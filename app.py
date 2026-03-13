import re
import math
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─────────────────────────────────────────────────────────────
# MODEL PARAMETERS  (3-variable logistic PD model)
# Fitted on World Bank Global Findex 2024 Kenya Microdata
# AUC = 0.7409
# ─────────────────────────────────────────────────────────────
DEFAULT_BETA0 =  4.8497
DEFAULT_BETA1 = -1.0054
DEFAULT_BETA2 = -0.1700
DEFAULT_BETA3 =  0.5851
MEAN_AGE      = 31.4
SD_AGE        = 12.4
MEAN_STABLE   = 0.22

# ─────────────────────────────────────────────────────────────
# CORE ACTUARIAL FUNCTIONS
# ─────────────────────────────────────────────────────────────
def logistic_pd(income_ksh, stable_job=MEAN_STABLE, age=MEAN_AGE,
                beta0=DEFAULT_BETA0, beta1=DEFAULT_BETA1,
                beta2=DEFAULT_BETA2, beta3=DEFAULT_BETA3):
    if income_ksh <= 0:
        return 1.0
    x      = math.log(income_ksh / 100.0)
    age_sc = (age - MEAN_AGE) / SD_AGE
    z      = beta0 + beta1*x + beta2*float(stable_job) + beta3*age_sc
    return float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z)))))

def annuity_factor(r, n):
    if n <= 0: return 0.0
    if abs(r) < 1e-12: return float(n)
    return float((1.0 - (1.0 + r)**(-n)) / r)

def fair_installment(cash_price, deposit_pct, admin_cost_pct, n_months, r_monthly, pd_est):
    op         = float(cash_price)
    deposit    = op * (deposit_pct / 100.0)
    admin_cost = op * (admin_cost_pct / 100.0)
    cf_revised = op + admin_cost - deposit
    af         = annuity_factor(float(r_monthly), int(n_months))
    repay_prob = max(1e-9, 1.0 - float(pd_est))
    m          = float("nan") if af <= 0 else cf_revised / (repay_prob * af)
    return {
        "op": op, "deposit_amount": deposit, "admin_cost_amount": admin_cost,
        "cf_revised": cf_revised, "annuity_factor": af, "repay_prob": repay_prob,
        "fair_monthly_installment": m,
        "fair_total_paid_if_no_default": deposit + (m * n_months),
        "radcf_present_value": deposit + (m * af * repay_prob),
    }

def implied_monthly_rate(P, payment, n):
    if P <= 0 or n <= 0 or payment * n < P: return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid   = (lo + hi) / 2.0
        denom = 1.0 - (1.0 + mid)**(-n)
        if denom <= 0: lo = mid; continue
        if P * mid / denom > payment: hi = mid
        else: lo = mid
    return (lo + hi) / 2.0

def effective_apr(i):
    return float("nan") if not np.isfinite(i) else float((1.0+i)**12 - 1.0)

def extract_deal_fields(text):
    t     = (text or "").lower().replace(",", " ")
    money = r"(?:ksh|kes)\s*([0-9]{2,})"
    pct   = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"
    cash_price = None
    m = re.search(r"(cash price|cash|price)\s*[:\-]?\s*" + money, t)
    if m: cash_price = float(m.group(2))
    if cash_price is None:
        m2 = re.search(money, t)
        if m2: cash_price = float(m2.group(1))
    deposit_pct, deposit_amount = None, None
    mdp = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + pct, t)
    if mdp: deposit_pct = float(mdp.group(2))
    mda = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + money, t)
    if mda: deposit_amount = float(mda.group(2))
    term_months = None
    mt = re.search(r"([0-9]{1,2})\s*(months|month|mos|mo)\b", t)
    if mt: term_months = int(mt.group(1))
    monthly_installment = None
    mm = re.search(r"(installment|instalment|monthly|per month)\s*[:\-]?\s*" + money, t)
    if mm: monthly_installment = float(mm.group(2))
    admin_pct, admin_amount = None, None
    mapct = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + pct, t)
    if mapct: admin_pct = float(mapct.group(3))
    maamt = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + money, t)
    if maamt: admin_amount = float(maamt.group(3))
    return {"cash_price": cash_price, "deposit_pct": deposit_pct,
            "deposit_amount": deposit_amount, "term_months": term_months,
            "monthly_installment": monthly_installment,
            "admin_pct": admin_pct, "admin_amount": admin_amount}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def pd_bucket(pd_val):
    if pd_val >= 0.50:
        return ("High", "#ef4444",
                "High repayment risk. Fair installments increase significantly to compensate expected default losses.")
    if pd_val >= 0.25:
        return ("Moderate", "#f59e0b",
                "Moderate repayment risk. Pricing includes a meaningful credit-risk adjustment.")
    return ("Low", "#22c55e",
            "Low repayment risk. Pricing requires a smaller credit-risk adjustment.")

def age_risk_label(age):
    if age < 26:  return "18–25 · Young borrower"
    if age < 36:  return "26–35 · Prime age, moderate risk"
    if age < 46:  return "36–45 · Mid-career, elevated risk"
    if age < 61:  return "46–60 · Senior borrower, high risk"
    return "60+ · Very high observed default rate"

def fairness_tag(over_pct):
    if not np.isfinite(over_pct): return ("", "#6b7280", "")
    if over_pct >= 0.25: return ("Severely Overpriced", "#ef4444",
                                  "Market pricing is far above RADCF fair value.")
    if over_pct >= 0.10: return ("Overpriced", "#f59e0b",
                                  "Market pricing is above RADCF fair value.")
    if over_pct >= -0.10: return ("Near Fair", "#22c55e",
                                   "Market pricing is close to RADCF fair value.")
    return ("Below Fair", "#3b82f6", "Market pricing is below RADCF fair value.")

def ksh(x):
    if x is None or not np.isfinite(x): return "—"
    return f"KSh {x:,.0f}"

def ksh2(x):
    if x is None or not np.isfinite(x): return "—"
    return f"KSh {x:,.2f}"

def run_scenarios(cash_price, deposit_pct, n_months, r_monthly,
                  pd_val, admin_cost_pct, stable_job, borrower_age,
                  beta0, beta1, beta2, beta3):
    pd_low    = max(0.0, pd_val * 0.9)
    pd_high   = min(1.0, pd_val * 1.1)
    pd_stable = logistic_pd(cash_price, 1, borrower_age, beta0, beta1, beta2, beta3)
    pd_unstab = logistic_pd(cash_price, 0, borrower_age, beta0, beta1, beta2, beta3)
    pd_stable = logistic_pd(float(cash_price) if cash_price else 25000,
                            1, borrower_age, beta0, beta1, beta2, beta3)
    pd_unstab = logistic_pd(float(cash_price) if cash_price else 25000,
                            0, borrower_age, beta0, beta1, beta2, beta3)
    # recalc properly using income
    pd_stable = logistic_pd(25000, 1, borrower_age, beta0, beta1, beta2, beta3)
    pd_unstab = logistic_pd(25000, 0, borrower_age, beta0, beta1, beta2, beta3)
    pd_young  = logistic_pd(25000, stable_job, 22, beta0, beta1, beta2, beta3)
    pd_older  = logistic_pd(25000, stable_job, 50, beta0, beta1, beta2, beta3)
    scenarios = [
        ("Base Case",          pd_val,    admin_cost_pct, r_monthly),
        ("PD −10%",            pd_low,    admin_cost_pct, r_monthly),
        ("PD +10%",            pd_high,   admin_cost_pct, r_monthly),
        ("Stable Income",      pd_stable, admin_cost_pct, r_monthly),
        ("Unstable Income",    pd_unstab, admin_cost_pct, r_monthly),
        ("Age 22 (Young)",     pd_young,  admin_cost_pct, r_monthly),
        ("Age 50 (Senior)",    pd_older,  admin_cost_pct, r_monthly),
        ("Admin Cost 8%",      pd_val,    8.0,            r_monthly),
        ("Rate +2pp",          pd_val,    admin_cost_pct, r_monthly + 0.02),
        ("Rate −1pp",          pd_val,    admin_cost_pct, max(0.0, r_monthly - 0.01)),
    ]
    rows = []
    for name, pd_s, adm, r_s in scenarios:
        rr = fair_installment(cash_price, deposit_pct, adm, int(n_months), r_s, pd_s)
        rows.append({
            "Scenario": name,
            "PD": round(float(pd_s), 3),
            "Admin%": float(adm),
            "r": round(float(r_s), 4),
            "Fair Monthly (KSh)": round(rr["fair_monthly_installment"], 2),
            "Fair Total (KSh)": round(rr["fair_total_paid_if_no_default"], 2),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────
def build_pdf_report(report_title, generated_dt, inputs, pd_params,
                     pd_value, radcf, market, sensitivity_df):
    DARK  = colors.HexColor("#0f172a")
    BLUE  = colors.HexColor("#1d4ed8")
    LBLUE = colors.HexColor("#eff6ff")
    GREY  = colors.HexColor("#f8fafc")
    MID   = colors.HexColor("#64748b")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm,
                            title="RADCF Fair Pricing Report")
    styles = getSampleStyleSheet()
    def S(name, **kw): return ParagraphStyle(name, **kw)
    title_s  = S("T", fontSize=20, textColor=DARK, fontName="Helvetica-Bold",
                 alignment=TA_LEFT, spaceAfter=4, leading=24)
    sub_s    = S("Sub", fontSize=10, textColor=MID, fontName="Helvetica",
                 alignment=TA_LEFT, spaceAfter=12)
    h2_s     = S("H2", fontSize=12, textColor=DARK, fontName="Helvetica-Bold",
                 spaceBefore=14, spaceAfter=6)
    body_s   = S("B", fontSize=9.5, textColor=DARK, fontName="Helvetica",
                 leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    small_s  = S("Sm", fontSize=8, textColor=MID, fontName="Helvetica", leading=12)
    formula_s = S("Fm", fontSize=9, textColor=DARK, fontName="Helvetica-Oblique",
                  leading=13, leftIndent=10)

    def make_table(data, col_widths, header=True):
        t = Table(data, colWidths=col_widths)
        style = [
            ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",  (0,0), (-1,-1), 9),
            ("GRID",      (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",(0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("LEFTPADDING",(0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0, 1 if header else 0),(-1,-1),[colors.white, GREY]),
        ]
        if header:
            style += [
                ("BACKGROUND",(0,0),(-1,0), BLUE),
                ("TEXTCOLOR", (0,0),(-1,0), colors.white),
                ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ]
        else:
            style += [("FONTNAME",(0,0),(0,-1),"Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    story = []

    # Header block
    hdr = Table([[Paragraph("RADCF Fair Pricing Report", title_s)],
                 [Paragraph(report_title, sub_s)],
                 [Paragraph(
                     f"Generated: {generated_dt.strftime('%d %B %Y, %H:%M')} EAT  ·  "
                     f"Report ID: RADCF-{generated_dt.strftime('%Y%m%d-%H%M%S')}",
                     small_s)]],
                colWidths=[17*cm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), LBLUE),
        ("TOPPADDING",(0,0),(-1,-1), 14),
        ("BOTTOMPADDING",(0,0),(-1,-1), 14),
        ("LEFTPADDING",(0,0),(-1,-1), 16),
        ("LINEBELOW",(0,-1),(-1,-1), 2, BLUE),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 14))

    # 1. Contract Inputs
    story.append(Paragraph("1. Contract Summary", h2_s))
    input_data = [["Parameter","Value"],
        ["Cash Price (OP)", ksh2(inputs["cash_price"])],
        ["Repayment Term",  f'{inputs["n_months"]} months'],
        ["Deposit",         f'{inputs["deposit_pct"]:.1f}%  →  {ksh2(radcf["deposit_amount"])}'],
        ["Admin Cost",      f'{inputs["admin_cost_pct"]:.1f}%  →  {ksh2(radcf["admin_cost_amount"])}'],
        ["Monthly Discount Rate",  f'{inputs["r_monthly"]:.4f}  ({inputs["r_monthly"]*100:.2f}%/month)'],
        ["Borrower Monthly Income", ksh2(inputs["income_ksh"])],
        ["Job Stability",   "Stable income" if inputs["stable_job"]==1 else "Unstable / Irregular income"],
        ["Borrower Age",    f'{inputs["age"]:.0f} years  ({age_risk_label(inputs["age"])})'],
    ]
    story.append(make_table(input_data, [7*cm, 10*cm]))
    story.append(Spacer(1, 10))

    # 2. PD Model
    story.append(Paragraph("2. Probability of Default (3-Variable Logistic Model)", h2_s))
    story.append(Paragraph(
        "PD = 1 / (1 + exp(−(β₀ + β₁·ln(Income/100) + β₂·StableJob + β₃·AgeScaled)))",
        formula_s))
    story.append(Spacer(1,4))
    story.append(Paragraph(
        "Fitted on World Bank Global Findex 2024 Kenya Microdata (n=934 credit buyers). AUC = 0.7409.",
        small_s))
    story.append(Spacer(1,6))
    lvl, _, pd_expl = pd_bucket(pd_value)
    age_sc = (inputs["age"] - MEAN_AGE) / SD_AGE
    pd_data = [["Parameter","Value","Interpretation"],
        ["β₀ (intercept)", f'{pd_params["beta0"]:.4f}', "Baseline log-odds"],
        ["β₁ (log income)", f'{pd_params["beta1"]:.4f}', "Higher income → lower PD"],
        ["β₂ (stable job)", f'{pd_params["beta2"]:.4f}', "Stable income → lower PD"],
        ["β₃ (age scaled)", f'{pd_params["beta3"]:.4f}', "Older age → higher PD"],
        ["Age (standardised)", f'{age_sc:.3f}', f'Age {inputs["age"]:.0f} yrs'],
        ["Job Stability", "Stable (1)" if inputs["stable_job"]==1 else "Unstable (0)", ""],
        ["Estimated PD", f'{pd_value:.4f}  ({pd_value*100:.1f}%)', f'{lvl} Risk — {pd_expl}'],
    ]
    story.append(make_table(pd_data, [5*cm, 3.5*cm, 8.5*cm]))
    story.append(Spacer(1, 10))

    # 3. RADCF Computation
    story.append(Paragraph("3. RADCF Pricing Computation", h2_s))
    lines = [
        f"CF_revised  =  OP + AdminCost − Deposit  =  {ksh2(radcf['op'])} + {ksh2(radcf['admin_cost_amount'])} − {ksh2(radcf['deposit_amount'])}  =  {ksh2(radcf['cf_revised'])}",
        f"AF  =  (1 − (1+r)^(−n)) / r,  r={inputs['r_monthly']:.4f},  n={inputs['n_months']}  →  AF ≈ {radcf['annuity_factor']:.4f}",
        f"M  =  CF_revised / ((1−PD) × AF)  =  {ksh2(radcf['cf_revised'])} / ({radcf['repay_prob']:.4f} × {radcf['annuity_factor']:.4f})  =  {ksh2(radcf['fair_monthly_installment'])}",
    ]
    for ln in lines:
        story.append(Paragraph(ln, formula_s))
        story.append(Spacer(1,3))
    story.append(Spacer(1,6))

    # 4. Fair Price Outputs
    story.append(Paragraph("4. Fair Price Outputs", h2_s))
    out_data = [["Output","Value"],
        ["Deposit",                          ksh2(radcf["deposit_amount"])],
        ["Fair Monthly Installment (M)",     ksh2(radcf["fair_monthly_installment"])],
        ["Fair Total Paid (Deposit + M×n)",  ksh2(radcf["fair_total_paid_if_no_default"])],
        ["RADCF Present Value",              ksh2(radcf["radcf_present_value"])],
    ]
    story.append(make_table(out_data, [9*cm, 8*cm]))
    story.append(Spacer(1, 10))

    # 5. Market Comparison
    story.append(Paragraph("5. Market Comparison", h2_s))
    if market.get("provided"):
        tag, _, tag_expl = fairness_tag(market.get("over_pct", float("nan")))
        mc_data = [["Metric","Value"],
            ["Market Monthly Installment",  ksh2(market.get("market_monthly", float("nan")))],
            ["Market Total Repayment",      ksh2(market.get("market_total", float("nan")))],
            ["RADCF Fair Total",            ksh2(radcf["fair_total_paid_if_no_default"])],
            ["Overpricing Amount",          ksh2(market.get("over_amt", float("nan")))],
            ["Overpricing (%)",             f'{market.get("over_pct",0)*100:.2f}%'],
            ["Fairness Score (0–100)",      f'{market.get("fairness_score",0):.1f}'],
            ["Assessment",                  f'{tag} — {tag_expl}'],
        ]
        story.append(make_table(mc_data, [9*cm, 8*cm]))
    else:
        story.append(Paragraph("No market comparison values were provided.", body_s))
    story.append(Spacer(1, 10))

    # 6. Sensitivity
    story.append(Paragraph("6. Sensitivity Analysis", h2_s))
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        cols = ["Scenario","PD","Admin%","r","Fair Monthly (KSh)","Fair Total (KSh)"]
        dfp  = sensitivity_df[cols].copy()
        tdata = [cols] + [list(r) for r in dfp.itertuples(index=False)]
        t5 = Table(tdata, colWidths=[4.5*cm,1.7*cm,1.8*cm,1.6*cm,3.7*cm,3.7*cm])
        t5.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), BLUE),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",  (0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",  (0,0),(-1,-1), 8),
            ("GRID",      (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, GREY]),
            ("ALIGN",     (1,1),(-1,-1), "RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(t5)
    story.append(Spacer(1, 10))

    # 7. Conclusion
    story.append(Paragraph("7. Conclusion & Recommendation", h2_s))
    lvl, _, _ = pd_bucket(pd_value)
    conclusion = (
        f"Based on the RADCF framework, the actuarially fair plan for this borrower is: "
        f"Deposit {ksh2(radcf['deposit_amount'])}, monthly installment "
        f"{ksh2(radcf['fair_monthly_installment'])}, fair total "
        f"{ksh2(radcf['fair_total_paid_if_no_default'])}. "
        f"The estimated probability of default is {pd_value*100:.1f}% ({lvl} risk), "
        f"reflecting income of {ksh2(inputs['income_ksh'])}/month, "
        f"{'stable' if inputs['stable_job']==1 else 'unstable'} income, "
        f"and age {inputs['age']:.0f} years. "
    )
    if market.get("provided"):
        tag, _, tag_expl = fairness_tag(market.get("over_pct", float("nan")))
        conclusion += (f"Market deal assessed as {tag} with "
                       f"{market.get('over_pct',0)*100:.2f}% overpricing vs RADCF fair value. ")
    conclusion += "Sensitivity results indicate which parameters most influence fair pricing."
    story.append(Paragraph(conclusion, body_s))
    story.append(Spacer(1, 10))

    # 8. Assumptions
    story.append(Paragraph("8. Assumptions & Limitations", h2_s))
    for a in [
        "PD model uses 3 predictors: income, job stability, and age. Fitted on Findex 2024 Kenya (n=934). AUC=0.7409.",
        "Job stability and age sourced from World Bank Global Findex 2024 Kenya dataset.",
        "Income distribution parameters (μ=5.1967, σ=0.7381) from Kenya Household Income Dataset (n=1,000).",
        "PD is assumed constant across the repayment term (simplifying assumption).",
        "No recovery after default assumed (Loss Given Default ≈ 100%).",
        "Market markup reference of 89% sourced from Citizen Digital (2025) and Business Daily Africa (2024).",
    ]:
        story.append(Paragraph(f"• {a}", body_s))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by RADCF Fair Pricing Engine · Egerton University · Department of Mathematics",
        small_s))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RADCF Fair Pricing Engine",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --navy:   #0f172a;
    --blue:   #1d4ed8;
    --lblue:  #3b82f6;
    --ice:    #eff6ff;
    --mint:   #22c55e;
    --amber:  #f59e0b;
    --red:    #ef4444;
    --slate:  #64748b;
    --border: #e2e8f0;
    --card:   #ffffff;
    --bg:     #f8fafc;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(34,197,94,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.25;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 0.92rem;
    color: #93c5fd;
    max-width: 680px;
    line-height: 1.6;
    margin-bottom: 1.4rem;
}
.hero-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.75rem; font-weight: 500;
    padding: 0.3rem 0.75rem;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    color: #e0f2fe;
    backdrop-filter: blur(4px);
}

/* ── Section headers ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--lblue);
    margin-bottom: 0.25rem;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--navy);
    margin-bottom: 1rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
.metric-card .accent-bar {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
}
.metric-label { font-size: 0.75rem; font-weight: 600; color: var(--slate);
                letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.35rem; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 1.5rem;
                font-weight: 500; color: var(--navy); line-height: 1.2; }
.metric-sub   { font-size: 0.78rem; color: var(--slate); margin-top: 0.3rem; }

/* ── PD gauge ── */
.pd-gauge-wrap { text-align: center; padding: 1rem 0; }
.pd-number { font-family: 'DM Mono', monospace; font-size: 3rem;
             font-weight: 500; line-height: 1; }
.pd-label  { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.06em;
             text-transform: uppercase; margin-top: 0.3rem; }

/* ── Formula box ── */
.formula-box {
    background: #0f172a;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #93c5fd;
    line-height: 2;
    margin: 0.75rem 0;
}
.formula-box .var  { color: #fbbf24; }
.formula-box .num  { color: #34d399; }
.formula-box .op   { color: #f472b6; }
.formula-box .res  { color: #ffffff; font-weight: 600; }

/* ── Overpricing banner ── */
.overpricing-banner {
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    border-left: 4px solid;
}
.banner-severe  { background: #fef2f2; border-color: #ef4444; }
.banner-over    { background: #fffbeb; border-color: #f59e0b; }
.banner-fair    { background: #f0fdf4; border-color: #22c55e; }
.banner-below   { background: #eff6ff; border-color: #3b82f6; }

/* ── Comparison bar ── */
.compare-bar { margin: 1rem 0; }
.compare-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
.compare-name { font-size: 0.8rem; font-weight: 600; width: 120px; color: var(--navy); }
.compare-track { flex: 1; background: var(--border); border-radius: 100px; height: 10px; overflow: hidden; }
.compare-fill  { height: 100%; border-radius: 100px; }
.compare-val   { font-family: 'DM Mono', monospace; font-size: 0.78rem;
                 color: var(--slate); min-width: 90px; text-align: right; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--slate);
}
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: white !important;
}

/* ── Input labels ── */
.stNumberInput label, .stRadio label,
.stSelectbox label { font-size: 0.82rem !important; font-weight: 600 !important;
                     color: var(--navy) !important; }

/* ── Dividers ── */
hr { border-color: var(--border); }

/* ── Step indicator ── */
.step-row { display: flex; gap: 0; margin-bottom: 2rem; }
.step-item { flex: 1; text-align: center; }
.step-circle {
    width: 36px; height: 36px; border-radius: 50%;
    background: var(--blue); color: white;
    font-weight: 700; font-size: 0.9rem;
    display: inline-flex; align-items: center; justify-content: center;
    margin-bottom: 0.4rem;
}
.step-text { font-size: 0.72rem; font-weight: 600; color: var(--slate);
             letter-spacing: 0.05em; text-transform: uppercase; }
.step-connector { flex: 1; height: 2px; background: var(--border);
                  margin-top: 18px; align-self: flex-start; }

/* ── Info boxes ── */
.info-box {
    background: var(--ice);
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.83rem;
    color: #1e40af;
    margin: 0.5rem 0;
}

/* ── Download button ── */
.stDownloadButton button {
    background: linear-gradient(135deg, var(--blue) 0%, #1e40af 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stDownloadButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.4) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--navy) !important;
    background: var(--bg) !important;
    border-radius: 8px !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }

/* ── Number input ── */
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border-color: var(--border) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">Actuarial Research Tool · Egerton University</div>
  <div class="hero-title">RADCF Fair Pricing Engine</div>
  <div class="hero-sub">
    Actuarial evaluation of consumer overpricing in Kenya's hire-purchase market.
    Computes risk-adjusted, actuarially fair smartphone installment plans using
    a 3-variable logistic default model.
  </div>
  <div class="hero-badges">
    <span class="badge">📊 3-Variable Logistic PD Model</span>
    <span class="badge">🎯 AUC = 0.7409</span>
    <span class="badge">📦 Findex 2024 Kenya · n=934</span>
    <span class="badge">🏛 Risk-Adjusted DCF Pricing</span>
    <span class="badge">📄 PDF Report Export</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── HOW IT WORKS (collapsible) ──────────────────────────────────────────────
with st.expander("📖 How It Works · Mathematical Framework", expanded=False):
    col_exp1, col_exp2 = st.columns([1.2, 0.8])
    with col_exp1:
        st.markdown("**Workflow**")
        st.markdown("""
1. **Estimate PD** via 3-variable logistic regression *(income, job stability, age)*
2. **Adjust repayments** using the repayment probability *(1 − PD)*
3. **Discount** expected cash flows at monthly rate *r* using the annuity factor
4. **Add** deposit and administrative costs to derive the fair total price
5. **Compare** against actual market prices to quantify overpricing
        """)
        st.markdown("**Model Coefficients (Findex 2024 Kenya)**")
        coef_df = pd.DataFrame({
            "Parameter": ["β₀ (intercept)", "β₁ (log income)", "β₂ (stable job)", "β₃ (age scaled)"],
            "Value":     [DEFAULT_BETA0, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_BETA3],
            "Effect":    ["Baseline", "↑ income → ↓ PD", "Stable job → ↓ PD", "↑ age → ↑ PD"],
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
    with col_exp2:
        st.markdown("**Core Formulas**")
        st.latex(r"PD = \frac{1}{1 + e^{-(\beta_0 + \beta_1\ln\frac{I}{100} + \beta_2 S + \beta_3 A)}}")
        st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
        st.latex(r"CF_{rev} = OP + \text{Admin} - \text{Deposit}")
        st.latex(r"M = \frac{CF_{rev}}{(1-PD)\cdot AF}")
        st.caption("Where I = monthly income, S = stable job (0/1), A = standardised age")

# ─── MAIN TABS ───────────────────────────────────────────────────────────────
tabs = st.tabs(["🧮  Manual Calculator", "📋  Paste Contract Text"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MANUAL CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:

    # Step indicators
    st.markdown("""
    <div class="step-row">
      <div class="step-item">
        <div class="step-circle">1</div>
        <div class="step-text">Contract Details</div>
      </div>
      <div class="step-connector"></div>
      <div class="step-item">
        <div class="step-circle">2</div>
        <div class="step-text">Borrower Profile</div>
      </div>
      <div class="step-connector"></div>
      <div class="step-item">
        <div class="step-circle">3</div>
        <div class="step-text">Pricing Results</div>
      </div>
      <div class="step-connector"></div>
      <div class="step-item">
        <div class="step-circle">4</div>
        <div class="step-text">Market Comparison</div>
      </div>
      <div class="step-connector"></div>
      <div class="step-item">
        <div class="step-circle">5</div>
        <div class="step-text">Sensitivity & Report</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT SECTION ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 1 & 2</div>'
                '<div class="section-title">Contract & Borrower Inputs</div>',
                unsafe_allow_html=True)

    colA, colB, colC = st.columns(3, gap="large")

    with colA:
        st.markdown("**📄 Contract Details**")
        cash_price = st.number_input(
            "Cash price (KSh)", min_value=0.0, value=25000.0, step=500.0, key="cp1",
            help="The retail cash price of the smartphone")
        n_months = st.number_input(
            "Repayment term (months)", min_value=1, value=12, step=1, key="n1")
        deposit_pct = st.number_input(
            "Deposit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="dp1",
            help="Upfront deposit as % of cash price. Typical: 30%")
        admin_cost_pct = st.number_input(
            "Administrative cost (%)", min_value=0.0, max_value=30.0, value=5.0,
            step=0.5, key="ad1", help="Processing/admin fee as % of cash price. Typical: 5%")

    with colB:
        st.markdown("**💰 Income & Rate**")
        income_ksh = st.number_input(
            "Borrower monthly income (KSh)", min_value=0.0, value=30000.0,
            step=1000.0, key="inc1")
        r_monthly = st.number_input(
            "Monthly discount rate r",
            min_value=0.0, value=round(0.13/12, 4),
            step=0.001, format="%.4f", key="r1",
            help="CBK rate 13% p.a. → 0.0108/month. Adjust for market risk premium.")
        st.markdown(f"""
        <div class="info-box">
        💡 At r = {r_monthly:.4f}/month → effective APR ≈ {((1+r_monthly)**12-1)*100:.1f}% p.a.<br>
        CF_revised = KSh {cash_price:,.0f} + {cash_price*admin_cost_pct/100:,.0f} − {cash_price*deposit_pct/100:,.0f}
        = <strong>KSh {cash_price + cash_price*admin_cost_pct/100 - cash_price*deposit_pct/100:,.0f}</strong>
        </div>""", unsafe_allow_html=True)

    with colC:
        st.markdown("**👤 Borrower Risk Profile**")
        borrower_age = st.number_input(
            "Borrower age (years)", min_value=18, max_value=90,
            value=31, step=1, key="age1")
        st.caption(f"🔍 {age_risk_label(borrower_age)}")
        stable_job = st.radio(
            "Income stability", options=[0,1],
            format_func=lambda x: "✅ Stable / Regular income" if x==1
                                  else "⚠️ Unstable / Irregular income",
            index=0, key="stab1",
            help=f"{MEAN_STABLE*100:.0f}% of Findex Kenya credit buyers have stable income")
        st.markdown("**⚙️ Advanced: Override PD Coefficients**")
        with st.expander("Override coefficients (optional)", expanded=False):
            beta0 = st.number_input("β₀", value=DEFAULT_BETA0, step=0.01, format="%.4f", key="b01")
            beta1 = st.number_input("β₁ (income)", value=DEFAULT_BETA1, step=0.01, format="%.4f", key="b11")
            beta2 = st.number_input("β₂ (stable job)", value=DEFAULT_BETA2, step=0.01, format="%.4f", key="b21")
            beta3 = st.number_input("β₃ (age scaled)", value=DEFAULT_BETA3, step=0.01, format="%.4f", key="b31")
        if "b01" not in st.session_state:
            beta0, beta1, beta2, beta3 = DEFAULT_BETA0, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_BETA3

    # ── COMPUTE ───────────────────────────────────────────────────────────────
    pd_val = logistic_pd(income_ksh, stable_job=stable_job, age=borrower_age,
                         beta0=beta0, beta1=beta1, beta2=beta2, beta3=beta3)
    res    = fair_installment(cash_price, deposit_pct, admin_cost_pct,
                              int(n_months), float(r_monthly), pd_val)
    lvl, lvl_color, lvl_expl = pd_bucket(pd_val)

    st.divider()

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 3</div>'
                '<div class="section-title">RADCF Fair Pricing Results</div>',
                unsafe_allow_html=True)

    # PD gauge + breakdown
    pd_col, metric_col = st.columns([0.38, 0.62], gap="large")

    with pd_col:
        # Plotly gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pd_val*100, 1),
            number={"suffix":"%","font":{"size":36,"family":"DM Mono","color":"#0f172a"}},
            title={"text": "Probability of Default","font":{"size":13,"color":"#64748b"}},
            gauge={
                "axis": {"range":[0,100],"tickwidth":1,"tickcolor":"#94a3b8",
                          "tickvals":[0,25,50,75,100]},
                "bar":  {"color": lvl_color,"thickness":0.3},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range":[0,25],  "color":"#dcfce7"},
                    {"range":[25,50], "color":"#fef9c3"},
                    {"range":[50,100],"color":"#fee2e2"},
                ],
                "threshold": {"line":{"color":"#0f172a","width":2},
                              "thickness":0.75,"value":pd_val*100},
            }
        ))
        fig_gauge.update_layout(
            height=240, margin=dict(l=20,r=20,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar":False})

        st.markdown(f"""
        <div style="background:{lvl_color}18;border:1px solid {lvl_color}44;
             border-radius:10px;padding:0.8rem 1rem;text-align:center">
          <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
               text-transform:uppercase;color:{lvl_color};margin-bottom:0.25rem">{lvl} Risk</div>
          <div style="font-size:0.78rem;color:#334155;line-height:1.5">{lvl_expl}</div>
        </div>""", unsafe_allow_html=True)

        # Stability impact
        pd_unstable = logistic_pd(income_ksh, 0, borrower_age, beta0, beta1, beta2, beta3)
        pd_stable_v = logistic_pd(income_ksh, 1, borrower_age, beta0, beta1, beta2, beta3)
        st.markdown(f"""
        <div style="margin-top:0.75rem;font-size:0.78rem;color:#475569">
          <strong>Stability Impact:</strong><br>
          Unstable income → PD {pd_unstable*100:.1f}%<br>
          Stable income &nbsp;&nbsp;→ PD {pd_stable_v*100:.1f}%
          &nbsp;(Δ {abs(pd_stable_v-pd_unstable)*100:.1f} pp)
        </div>""", unsafe_allow_html=True)

        with st.expander("PD component breakdown"):
            inc_c = beta1 * math.log(max(income_ksh,1) / 100)
            job_c = beta2 * float(stable_job)
            age_c = beta3 * (borrower_age - MEAN_AGE) / SD_AGE
            z     = beta0 + inc_c + job_c + age_c
            st.markdown(f"""
            <div class="formula-box">
<span class="var">β₀</span> (baseline) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="num">{beta0:+.4f}</span><br>
<span class="var">β₁</span> × ln(income/100) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="num">{inc_c:+.4f}</span><br>
<span class="var">β₂</span> × stable_job &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="num">{job_c:+.4f}</span><br>
<span class="var">β₃</span> × age_scaled &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="num">{age_c:+.4f}</span><br>
─────────────────────────────<br>
<span class="var">z</span> (linear combination) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="res">{z:+.4f}</span><br>
<span class="var">PD</span> = 1/(1+e^(-z)) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="res">{pd_val:.4f}</span>
            </div>""", unsafe_allow_html=True)

    with metric_col:
        # Main metric cards row 1
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="accent-bar" style="background:linear-gradient(90deg,#1d4ed8,#3b82f6)"></div>
              <div class="metric-label">Fair Monthly</div>
              <div class="metric-value">KSh {res['fair_monthly_installment']:,.0f}</div>
              <div class="metric-sub">per month for {n_months} months</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="accent-bar" style="background:linear-gradient(90deg,#0891b2,#06b6d4)"></div>
              <div class="metric-label">Deposit</div>
              <div class="metric-value">KSh {res['deposit_amount']:,.0f}</div>
              <div class="metric-sub">{deposit_pct:.0f}% of cash price</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="accent-bar" style="background:linear-gradient(90deg,#7c3aed,#a78bfa)"></div>
              <div class="metric-label">Admin Cost</div>
              <div class="metric-value">KSh {res['admin_cost_amount']:,.0f}</div>
              <div class="metric-sub">{admin_cost_pct:.0f}% of cash price</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        m4, m5 = st.columns(2)
        with m4:
            markup = (res['fair_total_paid_if_no_default'] - cash_price) / cash_price * 100
            st.markdown(f"""
            <div class="metric-card">
              <div class="accent-bar" style="background:linear-gradient(90deg,#059669,#34d399)"></div>
              <div class="metric-label">Fair Total Paid</div>
              <div class="metric-value">KSh {res['fair_total_paid_if_no_default']:,.0f}</div>
              <div class="metric-sub">+{markup:.1f}% over cash price · RADCF justified markup</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
              <div class="accent-bar" style="background:linear-gradient(90deg,#dc2626,#f87171)"></div>
              <div class="metric-label">RADCF Present Value</div>
              <div class="metric-value">KSh {res['radcf_present_value']:,.0f}</div>
              <div class="metric-sub">Expected PV of repayment stream</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        # RADCF formula breakdown
        st.markdown(f"""
        <div class="formula-box">
<span class="var">CF_rev</span> = {ksh(cash_price)} <span class="op">+</span> {ksh(res['admin_cost_amount'])} <span class="op">−</span> {ksh(res['deposit_amount'])} <span class="op">=</span> <span class="res">{ksh(res['cf_revised'])}</span><br>
<span class="var">AF</span> &nbsp;&nbsp;&nbsp;&nbsp;= (1 − (1+{r_monthly:.4f})^(-{n_months})) / {r_monthly:.4f} = <span class="res">{res['annuity_factor']:.4f}</span><br>
<span class="var">M</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {ksh(res['cf_revised'])} / ({res['repay_prob']:.4f} × {res['annuity_factor']:.4f}) = <span class="res">{ksh(res['fair_monthly_installment'])}</span>
        </div>""", unsafe_allow_html=True)

        # Affordability check
        if income_ksh > 0:
            iti = res['fair_monthly_installment'] / income_ksh * 100
            iti_color = "#ef4444" if iti > 40 else "#f59e0b" if iti > 20 else "#22c55e"
            iti_label = "High burden (>40%)" if iti > 40 else "Moderate (20–40%)" if iti > 20 else "Affordable (<20%)"
            fill_pct = min(100, iti)
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.2rem;margin-top:0.25rem">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                <span style="font-size:0.78rem;font-weight:700;color:#334155;text-transform:uppercase;letter-spacing:0.06em">Affordability (ITI Ratio)</span>
                <span style="font-family:'DM Mono',monospace;font-size:1rem;font-weight:600;color:{iti_color}">{iti:.1f}%</span>
              </div>
              <div style="background:#e2e8f0;border-radius:100px;height:8px;overflow:hidden">
                <div style="background:{iti_color};width:{fill_pct}%;height:100%;border-radius:100px;transition:width 0.4s"></div>
              </div>
              <div style="margin-top:0.4rem;font-size:0.75rem;color:{iti_color};font-weight:600">{iti_label}</div>
              <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem">
                Fair installment as % of KSh {income_ksh:,.0f}/month income · 30% threshold recommended
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── MARKET COMPARISON ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 4</div>'
                '<div class="section-title">Market Deal Comparison</div>',
                unsafe_allow_html=True)

    cmp_l, cmp_r = st.columns([0.45, 0.55], gap="large")
    market_info = {"provided": False}

    with cmp_l:
        st.markdown("**Enter actual market pricing to assess overpricing:**")
        market_monthly = st.number_input(
            "Market monthly installment (KSh)", min_value=0.0,
            value=0.0, step=100.0, key="m_m1",
            help="The actual installment charged by M-Kopa, Watu Credit, etc.")
        market_total_input = st.number_input(
            "Market total repayment (KSh)", min_value=0.0,
            value=0.0, step=500.0, key="m_t1",
            help="Total amount paid over the contract. If provided, takes precedence.")
        st.caption("💡 Leave at 0 to skip the market comparison.")

        # Quick presets
        st.markdown("**Quick presets (common providers):**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("M-Kopa\n3,150", key="p_mkopa", use_container_width=True):
            st.session_state["m_m1"] = 3150.0
            st.rerun()
        if preset_cols[1].button("Watu\n3,500", key="p_watu", use_container_width=True):
            st.session_state["m_m1"] = 3500.0
            st.rerun()
        if preset_cols[2].button("Lipa Mdogo\n2,800", key="p_lipa", use_container_width=True):
            st.session_state["m_m1"] = 2800.0
            st.rerun()

    with cmp_r:
        fair_total   = res["fair_total_paid_if_no_default"]
        over_amt     = float("nan")
        over_pct     = float("nan")
        implied_apr  = float("nan")
        mkt_total_used = 0.0

        if market_total_input > 0:
            mkt_total_used = float(market_total_input)
            over_amt = mkt_total_used - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")
        elif market_monthly > 0:
            mkt_total_used = res["deposit_amount"] + float(market_monthly)*int(n_months)
            over_amt       = mkt_total_used - fair_total
            over_pct       = (over_amt / fair_total) if fair_total > 0 else float("nan")
            principal      = cash_price - res["deposit_amount"]
            im             = implied_monthly_rate(principal, float(market_monthly), int(n_months))
            implied_apr    = effective_apr(im)

        if np.isfinite(over_pct):
            tag, tag_color, tag_expl = fairness_tag(over_pct)
            fairness_score = max(0.0, min(100.0, 100.0 - over_pct * 100.0))
            banner_cls = ("banner-severe" if over_pct >= 0.25 else
                          "banner-over"   if over_pct >= 0.10 else
                          "banner-fair"   if over_pct >= -0.1 else "banner-below")

            st.markdown(f"""
            <div class="overpricing-banner {banner_cls}">
              <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                   letter-spacing:0.1em;color:{tag_color};margin-bottom:0.4rem">{tag}</div>
              <div style="font-size:0.82rem;color:#334155">{tag_expl}</div>
            </div>""", unsafe_allow_html=True)

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown(f"""<div class="metric-card">
                  <div class="accent-bar" style="background:{tag_color}"></div>
                  <div class="metric-label">Overpricing</div>
                  <div class="metric-value" style="color:{tag_color}">{over_pct*100:.1f}%</div>
                  <div class="metric-sub">vs RADCF fair value</div>
                </div>""", unsafe_allow_html=True)
            with oc2:
                st.markdown(f"""<div class="metric-card">
                  <div class="accent-bar" style="background:#ef4444"></div>
                  <div class="metric-label">Extra Paid</div>
                  <div class="metric-value" style="font-size:1.15rem">KSh {over_amt:,.0f}</div>
                  <div class="metric-sub">annually above fair price</div>
                </div>""", unsafe_allow_html=True)
            with oc3:
                st.markdown(f"""<div class="metric-card">
                  <div class="accent-bar" style="background:#7c3aed"></div>
                  <div class="metric-label">Fairness Score</div>
                  <div class="metric-value">{fairness_score:.0f}<span style="font-size:1rem">/100</span></div>
                  <div class="metric-sub">0 = worst · 100 = perfect</div>
                </div>""", unsafe_allow_html=True)

            # Visual bar comparison
            max_val = max(fair_total, mkt_total_used)
            fair_pct_bar = fair_total / max_val * 100
            mkt_pct_bar  = mkt_total_used / max_val * 100

            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                 padding:1rem 1.25rem;margin-top:0.5rem">
              <div style="font-size:0.75rem;font-weight:700;color:#334155;
                   text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem">Total Repayment Comparison</div>
              <div class="compare-row">
                <div class="compare-name">RADCF Fair</div>
                <div class="compare-track">
                  <div class="compare-fill" style="width:{fair_pct_bar:.1f}%;background:#22c55e"></div>
                </div>
                <div class="compare-val">KSh {fair_total:,.0f}</div>
              </div>
              <div class="compare-row">
                <div class="compare-name">Market Price</div>
                <div class="compare-track">
                  <div class="compare-fill" style="width:{mkt_pct_bar:.1f}%;background:{tag_color}"></div>
                </div>
                <div class="compare-val">KSh {mkt_total_used:,.0f}</div>
              </div>
              <div class="compare-row">
                <div class="compare-name">Cash Price</div>
                <div class="compare-track">
                  <div class="compare-fill" style="width:{cash_price/max_val*100:.1f}%;background:#94a3b8"></div>
                </div>
                <div class="compare-val">KSh {cash_price:,.0f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if np.isfinite(implied_apr):
                st.markdown(f"""
                <div class="info-box" style="margin-top:0.5rem">
                📈 <strong>Implied effective APR</strong> on market deal: <strong>{implied_apr*100:.1f}%</strong>
                &nbsp;·&nbsp; vs CBK reference rate {r_monthly*12*100:.1f}%
                </div>""", unsafe_allow_html=True)

            market_info = {
                "provided":       True,
                "market_monthly": float(market_monthly) if market_monthly > 0 else float("nan"),
                "market_total":   float(mkt_total_used),
                "over_amt":       float(over_amt),
                "over_pct":       float(over_pct),
                "fairness_score": float(fairness_score),
                "tag":            tag,
                "tag_explain":    tag_expl,
                "implied_apr":    float(implied_apr),
            }
        else:
            st.markdown("""
            <div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
                 padding:2rem;text-align:center;color:#94a3b8">
              <div style="font-size:2rem;margin-bottom:0.5rem">📊</div>
              <div style="font-size:0.88rem;font-weight:500">
                Enter a market installment or total repayment on the left to compare
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Step 5</div>'
                '<div class="section-title">Sensitivity Analysis & Stress Testing</div>',
                unsafe_allow_html=True)

    sens_df = run_scenarios(
        cash_price, deposit_pct, n_months, r_monthly,
        pd_val, admin_cost_pct, stable_job, borrower_age,
        beta0, beta1, beta2, beta3)

    sa_l, sa_r = st.columns([0.48, 0.52], gap="large")
    with sa_l:
        st.markdown("**Scenario Table**")
        st.dataframe(
            sens_df.style.format({
                "PD": "{:.3f}",
                "Admin%": "{:.1f}",
                "r": "{:.4f}",
                "Fair Monthly (KSh)": "{:,.0f}",
                "Fair Total (KSh)": "{:,.0f}",
            }).background_gradient(
                subset=["Fair Monthly (KSh)","Fair Total (KSh)"],
                cmap="Blues"
            ),
            use_container_width=True, hide_index=True)

    with sa_r:
        metric = st.radio("Tornado chart metric:",
                          ["Fair Total (KSh)", "Fair Monthly (KSh)"],
                          horizontal=True, key="t_metric")
        base_row = sens_df[sens_df["Scenario"] == "Base Case"]
        if not base_row.empty:
            base_val = float(base_row.iloc[0][metric])
            plot_df  = sens_df[sens_df["Scenario"] != "Base Case"].copy()
            plot_df["Impact"]    = plot_df[metric].astype(float) - base_val
            plot_df["AbsImpact"] = plot_df["Impact"].abs()
            plot_df = plot_df.sort_values("AbsImpact", ascending=True)

            colors_list = ["#3b82f6" if v < 0 else "#ef4444" for v in plot_df["Impact"]]
            fig_t = go.Figure(go.Bar(
                y=plot_df["Scenario"],
                x=plot_df["Impact"],
                orientation="h",
                marker_color=colors_list,
                text=[f"KSh {v:+,.0f}" for v in plot_df["Impact"]],
                textposition="outside",
                textfont=dict(size=10, family="DM Mono"),
            ))
            fig_t.add_vline(x=0, line_width=1.5, line_color="#0f172a")
            fig_t.update_layout(
                height=380,
                xaxis_title=f"Δ {metric} vs Base (KSh {base_val:,.0f})",
                yaxis_title=None,
                title=dict(text="Tornado Chart — Sensitivity Impact",
                           font=dict(size=13,family="DM Sans"),x=0),
                margin=dict(l=10,r=80,t=45,b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(tickfont=dict(size=11,family="DM Sans")),
                xaxis=dict(gridcolor="#f1f5f9",zerolinecolor="#94a3b8"),
            )
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})
            most = plot_df.iloc[-1]
            st.caption(f"Most sensitive factor: **{most['Scenario']}** "
                       f"→ KSh {most['Impact']:+,.0f} vs base")

    st.divider()

    # ── PDF DOWNLOAD ──────────────────────────────────────────────────────────
    st.markdown("### 📄 Download Full Report")
    dl_col, info_col = st.columns([0.3, 0.7])
    with dl_col:
        pdf_bytes = build_pdf_report(
            report_title="Actuarial Evaluation of Consumer Overpricing in Kenya's Hire-Purchase Market",
            generated_dt=datetime.now(),
            inputs={
                "cash_price": cash_price, "n_months": int(n_months),
                "deposit_pct": float(deposit_pct), "admin_cost_pct": float(admin_cost_pct),
                "r_monthly": float(r_monthly), "income_ksh": float(income_ksh),
                "stable_job": int(stable_job), "age": float(borrower_age),
            },
            pd_params={"beta0":beta0,"beta1":beta1,"beta2":beta2,"beta3":beta3},
            pd_value=float(pd_val), radcf=res, market=market_info,
            sensitivity_df=sens_df,
        )
        st.download_button(
            label="⬇  Download RADCF Report (PDF)",
            data=pdf_bytes,
            file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", key="dl_pdf_manual",
            use_container_width=True,
        )
    with info_col:
        st.markdown("""
        <div class="info-box">
        The PDF report includes: contract summary, risk model parameters, RADCF computation,
        fair price outputs, market comparison (if provided), full sensitivity table,
        conclusion and recommendations, and assumptions/limitations.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PASTE CONTRACT TEXT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-label">Smart Extraction</div>'
                '<div class="section-title">Paste Contract or Offer Text</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📋 Paste a hire-purchase offer — WhatsApp message, SMS, advert text, or contract snippet.
    The engine will extract key fields and compute the RADCF fair price automatically.
    </div>""", unsafe_allow_html=True)

    sample = ("Cash price: KSh 25000. Deposit 30%. "
              "Pay KES 2500 per month for 12 months. Admin fee 5%.")
    txt = st.text_area("Paste text here", value=sample, height=130, key="txt_offer",
                       placeholder="e.g. Cash price: KSh 18000. Deposit 30%. "
                                   "Monthly installment KES 3150 for 12 months. Admin 5%.")

    extracted = extract_deal_fields(txt)

    # Show extraction result
    ext_cols = st.columns(4)
    fields = [
        ("Cash Price", f"KSh {extracted['cash_price']:,.0f}" if extracted['cash_price'] else "Not found", "💰"),
        ("Term", f"{extracted['term_months']} months" if extracted['term_months'] else "Not found", "📅"),
        ("Deposit", f"{extracted['deposit_pct']}%" if extracted['deposit_pct'] else
                    (f"KSh {extracted['deposit_amount']:,.0f}" if extracted['deposit_amount'] else "Not found"), "🏦"),
        ("Admin Fee", f"{extracted['admin_pct']}%" if extracted['admin_pct'] else "Not found", "⚙️"),
    ]
    for col, (label, val, icon) in zip(ext_cols, fields):
        found = val != "Not found"
        col.markdown(f"""
        <div style="background:{'#f0fdf4' if found else '#fef2f2'};
             border:1px solid {'#bbf7d0' if found else '#fecaca'};
             border-radius:10px;padding:0.75rem;text-align:center">
          <div style="font-size:1.3rem">{icon}</div>
          <div style="font-size:0.68rem;font-weight:700;color:#64748b;
               text-transform:uppercase;letter-spacing:0.08em">{label}</div>
          <div style="font-size:0.85rem;font-weight:600;
               color:{'#166534' if found else '#991b1b'};margin-top:0.2rem">{val}</div>
        </div>""", unsafe_allow_html=True)

    if extracted.get("monthly_installment"):
        st.markdown(f"""
        <div class="info-box" style="margin-top:0.75rem">
        🔍 Detected market installment: <strong>KSh {extracted['monthly_installment']:,.0f}/month</strong>
        — this will be used for overpricing comparison below.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Adjust extracted values if needed:**")

    colX, colY, colZ = st.columns(3, gap="large")
    with colX:
        cash_price2 = st.number_input(
            "Cash price (KSh)", min_value=0.0,
            value=float(extracted["cash_price"] or 25000.0), step=500.0, key="cp2")
        n_months2 = st.number_input(
            "Repayment term (months)", min_value=1,
            value=int(extracted["term_months"] or 12), step=1, key="n2")

    with colY:
        dep_pct_g = extracted["deposit_pct"]
        if dep_pct_g is None and extracted["deposit_amount"] and cash_price2 > 0:
            dep_pct_g = 100.0 * float(extracted["deposit_amount"]) / float(cash_price2)
        deposit_pct2 = st.number_input(
            "Deposit (%)", min_value=0.0, max_value=100.0,
            value=float(dep_pct_g or 30.0), step=1.0, key="dp2")
        adm_g = extracted["admin_pct"]
        if adm_g is None and extracted["admin_amount"] and cash_price2 > 0:
            adm_g = 100.0 * float(extracted["admin_amount"]) / float(cash_price2)
        admin2 = st.number_input(
            "Administrative cost (%)", min_value=0.0, max_value=30.0,
            value=float(adm_g or 5.0), step=0.5, key="ad2")
        r2 = st.number_input(
            "Monthly discount rate r", min_value=0.0,
            value=round(0.13/12, 4), step=0.001, format="%.4f", key="r2")

    with colZ:
        income2 = st.number_input(
            "Borrower monthly income (KSh)", min_value=0.0,
            value=30000.0, step=1000.0, key="inc2")
        age2 = st.number_input(
            "Borrower age (years)", min_value=18, max_value=90,
            value=31, step=1, key="age2")
        stable2 = st.radio(
            "Income stability", options=[0,1],
            format_func=lambda x: "✅ Stable" if x==1 else "⚠️ Unstable",
            index=0, key="stab2")
        st.caption(f"🔍 {age_risk_label(age2)}")

    pd2  = logistic_pd(income2, stable_job=stable2, age=age2)
    res2 = fair_installment(cash_price2, deposit_pct2, admin2,
                            int(n_months2), float(r2), pd2)
    lvl2, lvl_col2, _ = pd_bucket(pd2)

    st.divider()
    r1, r2c, r3, r4 = st.columns(4)
    r1.markdown(f"""<div class="metric-card">
      <div class="accent-bar" style="background:{lvl_col2}"></div>
      <div class="metric-label">Probability of Default</div>
      <div class="metric-value">{pd2*100:.1f}%</div>
      <div class="metric-sub">{lvl2} risk</div>
    </div>""", unsafe_allow_html=True)
    r2c.markdown(f"""<div class="metric-card">
      <div class="accent-bar" style="background:#1d4ed8"></div>
      <div class="metric-label">Fair Monthly Installment</div>
      <div class="metric-value">KSh {res2['fair_monthly_installment']:,.0f}</div>
      <div class="metric-sub">per month</div>
    </div>""", unsafe_allow_html=True)
    r3.markdown(f"""<div class="metric-card">
      <div class="accent-bar" style="background:#059669"></div>
      <div class="metric-label">Fair Total Paid</div>
      <div class="metric-value">KSh {res2['fair_total_paid_if_no_default']:,.0f}</div>
      <div class="metric-sub">deposit + installments</div>
    </div>""", unsafe_allow_html=True)

    # Market comparison from extracted installment
    mkt_inst = extracted.get("monthly_installment")
    if mkt_inst:
        mkt_tot2   = res2["deposit_amount"] + mkt_inst * int(n_months2)
        over_amt2  = mkt_tot2 - res2["fair_total_paid_if_no_default"]
        over_pct2  = over_amt2 / res2["fair_total_paid_if_no_default"]
        tag2, tc2, _ = fairness_tag(over_pct2)
        r4.markdown(f"""<div class="metric-card">
          <div class="accent-bar" style="background:{tc2}"></div>
          <div class="metric-label">Market Overpricing</div>
          <div class="metric-value" style="color:{tc2}">{over_pct2*100:.1f}%</div>
          <div class="metric-sub">{tag2}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # PDF for tab 2
    pd_low2  = max(0.0, pd2*0.9);  pd_high2 = min(1.0, pd2*1.1)
    pd_s2    = logistic_pd(income2, 1, age2);  pd_u2 = logistic_pd(income2, 0, age2)
    scen2    = [("Base Case",pd2,admin2,r2),("PD −10%",pd_low2,admin2,r2),
                ("PD +10%",pd_high2,admin2,r2),("Stable Income",pd_s2,admin2,r2),
                ("Unstable Income",pd_u2,admin2,r2),("Admin 8%",pd2,8.0,r2),
                ("Rate +2pp",pd2,admin2,r2+0.02),("Rate −1pp",pd2,admin2,max(0,r2-0.01))]
    rows2 = []
    for name,pd_s,adm,r_s in scen2:
        rr = fair_installment(cash_price2,deposit_pct2,adm,int(n_months2),r_s,pd_s)
        rows2.append({"Scenario":name,"PD":round(pd_s,3),"Admin%":adm,
                      "r":round(r_s,4),
                      "Fair Monthly (KSh)":round(rr["fair_monthly_installment"],2),
                      "Fair Total (KSh)":round(rr["fair_total_paid_if_no_default"],2)})
    sens2 = pd.DataFrame(rows2)

    mkt_info2 = {"provided": False}
    if mkt_inst and np.isfinite(over_pct2):
        mkt_info2 = {"provided":True,"market_monthly":mkt_inst,
                     "market_total":mkt_tot2,"over_amt":over_amt2,
                     "over_pct":over_pct2,"fairness_score":max(0,min(100,100-over_pct2*100)),
                     "tag":tag2,"tag_explain":"Auto-detected from pasted text","implied_apr":float("nan")}

    dl2_col, info2_col = st.columns([0.3, 0.7])
    with dl2_col:
        pdf2 = build_pdf_report(
            "Actuarial Evaluation of Consumer Overpricing in Kenya's Hire-Purchase Market",
            datetime.now(),
            {"cash_price":float(cash_price2),"n_months":int(n_months2),
             "deposit_pct":float(deposit_pct2),"admin_cost_pct":float(admin2),
             "r_monthly":float(r2),"income_ksh":float(income2),
             "stable_job":int(stable2),"age":float(age2)},
            {"beta0":DEFAULT_BETA0,"beta1":DEFAULT_BETA1,
             "beta2":DEFAULT_BETA2,"beta3":DEFAULT_BETA3},
            float(pd2), res2, mkt_info2, sens2,
        )
        st.download_button(
            "⬇  Download RADCF Report (PDF)", data=pdf2,
            file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", key="dl_pdf_auto",
            use_container_width=True,
        )
    with info2_col:
        st.markdown("""
        <div class="info-box">
        Text extraction uses regex pattern matching. Review extracted values carefully
        before running computations. For best results, include keywords like
        <em>"cash price: KSh …"</em>, <em>"deposit X%"</em>, <em>"X months"</em>.
        </div>""", unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid #e2e8f0;
     text-align:center;color:#94a3b8;font-size:0.78rem">
  <strong style="color:#64748b">RADCF Fair Pricing Engine</strong> ·
  Egerton University · Department of Mathematics · Actuarial Science ·
  Model: World Bank Findex 2024 Kenya (AUC 0.7409) ·
  <em>For research and educational purposes</em>
</div>""", unsafe_allow_html=True)
