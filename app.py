import re
import math
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ═══════════════════════════════════════════════════════════════════
# MODEL PARAMETERS — exactly as in the research paper
# Data: KNBS KCHSP 2022 (n=987 households after cleaning)
# Model: Single-variable logistic regression (income only)
# β₀=1.5309  z=4.807   p=5.11×10⁻⁷
# β₁=−0.7322 z=−6.963  p=3.33×10⁻⁹
# ═══════════════════════════════════════════════════════════════════
BETA0 = 1.5309
BETA1 = -0.7322

INCOME_BANDS = [
    ("Below KES 10,000",      7_500,  0.423),
    ("KES 10,000 – 15,000",  12_500,  0.382),
    ("KES 15,001 – 25,000",  20_000,  0.351),
    ("KES 25,001 – 40,000",  32_500,  0.321),
    ("KES 40,001 – 70,000",  55_000,  0.286),
    ("Above KES 70,000",     85_000,  0.147),
]
MARKET_INSTALLMENT = 3150.0  # KES — M-Kopa / Watu Credit benchmark


# ═══════════════════════════════════════════════════════════════════
# CORE ACTUARIAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def logistic_pd(income_ksh: float,
                beta0: float = BETA0,
                beta1: float = BETA1) -> float:
    """PD = 1 / (1 + exp(−(β₀ + β₁·ln(Income/1000))))"""
    if income_ksh <= 0:
        return 1.0
    z = beta0 + beta1 * math.log(income_ksh / 1000.0)
    return float(max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-z)))))


def annuity_factor(r: float, n: int) -> float:
    """AF = (1 − (1+r)^−n) / r"""
    if n <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return float(n)
    return float((1.0 - (1.0 + r) ** (-n)) / r)


def fair_installment(cash_price: float, deposit_pct: float,
                     admin_cost_pct: float, n_months: int,
                     r_monthly: float, pd_est: float) -> dict:
    op         = float(cash_price)
    deposit    = op * (deposit_pct / 100.0)
    admin_cost = op * (admin_cost_pct / 100.0)
    cf_revised = op + admin_cost - deposit
    af         = annuity_factor(float(r_monthly), int(n_months))
    repay_prob = max(1e-9, 1.0 - float(pd_est))
    m          = float("nan") if af <= 0 else cf_revised / (repay_prob * af)
    return {
        "op":             op,
        "deposit":        deposit,
        "admin_cost":     admin_cost,
        "cf_revised":     cf_revised,
        "af":             af,
        "repay_prob":     repay_prob,
        "fair_monthly":   m,
        "fair_total":     deposit + (m * n_months),
        "radcf_pv":       deposit + (m * af * repay_prob),
    }


def implied_monthly_rate(P: float, payment: float, n: int) -> float:
    if P <= 0 or n <= 0 or payment * n < P:
        return float("nan")
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid   = (lo + hi) / 2.0
        denom = 1.0 - (1.0 + mid) ** (-n)
        if denom <= 0:
            lo = mid
            continue
        if P * mid / denom > payment:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def effective_apr(i: float) -> float:
    return float("nan") if not np.isfinite(i) else float((1.0 + i) ** 12 - 1.0)


def extract_deal_fields(text: str) -> dict:
    t     = (text or "").lower().replace(",", " ")
    money = r"(?:ksh|kes)\s*([0-9]{2,})"
    pct   = r"([0-9]{1,2}(?:\.[0-9]+)?)\s*%"
    cash_price = None
    m = re.search(r"(cash price|cash|price)\s*[:\-]?\s*" + money, t)
    if m:
        cash_price = float(m.group(2))
    if cash_price is None:
        m2 = re.search(money, t)
        if m2:
            cash_price = float(m2.group(1))
    deposit_pct, deposit_amount = None, None
    mdp = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + pct, t)
    if mdp:
        deposit_pct = float(mdp.group(2))
    mda = re.search(r"(deposit|downpayment|down payment)\s*[:\-]?\s*" + money, t)
    if mda:
        deposit_amount = float(mda.group(2))
    term_months = None
    mt = re.search(r"([0-9]{1,2})\s*(months|month|mos|mo)\b", t)
    if mt:
        term_months = int(mt.group(1))
    monthly_installment = None
    mm = re.search(r"(installment|instalment|monthly|per month)\s*[:\-]?\s*" + money, t)
    if mm:
        monthly_installment = float(mm.group(2))
    admin_pct, admin_amount = None, None
    mapct = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + pct, t)
    if mapct:
        admin_pct = float(mapct.group(3))
    maamt = re.search(r"(admin|administration|processing)\s*(fee|cost)?\s*[:\-]?\s*" + money, t)
    if maamt:
        admin_amount = float(maamt.group(3))
    return {
        "cash_price": cash_price, "deposit_pct": deposit_pct,
        "deposit_amount": deposit_amount, "term_months": term_months,
        "monthly_installment": monthly_installment,
        "admin_pct": admin_pct, "admin_amount": admin_amount,
    }


# ─── Helpers ────────────────────────────────────────────────────────
def pd_bucket(pd_val: float):
    if pd_val >= 0.50:
        return ("High", "#ef4444",
                "High repayment risk. Fair installment increases significantly to compensate expected losses.")
    if pd_val >= 0.25:
        return ("Moderate", "#f59e0b",
                "Moderate repayment risk. Pricing includes a meaningful credit-risk adjustment.")
    return ("Low", "#22c55e",
            "Low repayment risk. Only a small credit-risk adjustment is required.")


def fairness_tag(over_pct: float):
    if not np.isfinite(over_pct):
        return ("", "#6b7280", "")
    if over_pct >= 0.25:
        return ("Severely Overpriced", "#ef4444", "Market pricing is far above RADCF fair value.")
    if over_pct >= 0.10:
        return ("Overpriced", "#f59e0b", "Market pricing is above RADCF fair value.")
    if over_pct >= -0.10:
        return ("Near Fair", "#22c55e", "Market pricing is close to RADCF fair value.")
    return ("Below Fair", "#3b82f6", "Market pricing is below RADCF fair value.")


def ksh(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"KSh {x:,.0f}"


def run_scenarios(cash_price, deposit_pct, n_months, r_monthly, pd_val, admin_pct):
    rows = []
    for name, pd_s, adm, r_s, n_s in [
        ("Base Case",       pd_val,              admin_pct, r_monthly,       n_months),
        ("PD −10%",         max(0, pd_val*0.90), admin_pct, r_monthly,       n_months),
        ("PD +10%",         min(1, pd_val*1.10), admin_pct, r_monthly,       n_months),
        ("Admin Cost 8%",   pd_val,              8.0,       r_monthly,       n_months),
        ("Admin Cost 3%",   pd_val,              3.0,       r_monthly,       n_months),
        ("Rate +2 pp",      pd_val,              admin_pct, r_monthly+0.02,  n_months),
        ("Rate −1 pp",      pd_val,              admin_pct, max(0,r_monthly-0.01), n_months),
        ("Term 18 months",  pd_val,              admin_pct, r_monthly,       18),
        ("Term 6 months",   pd_val,              admin_pct, r_monthly,       6),
    ]:
        rr = fair_installment(cash_price, deposit_pct, adm, int(n_s), r_s, pd_s)
        rows.append({
            "Scenario":           name,
            "PD":                 round(float(pd_s), 3),
            "Admin %":            float(adm),
            "r":                  round(float(r_s), 4),
            "Term (mo)":          int(n_s),
            "Fair Monthly (KSh)": round(rr["fair_monthly"], 2),
            "Fair Total (KSh)":   round(rr["fair_total"], 2),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════
def build_pdf(inputs: dict, pd_value: float, radcf: dict,
              market: dict, sensitivity_df: pd.DataFrame) -> bytes:
    DARK  = colors.HexColor("#0f172a")
    BLUE  = colors.HexColor("#1d4ed8")
    LBLUE = colors.HexColor("#eff6ff")
    BGREY = colors.HexColor("#f8fafc")
    MID   = colors.HexColor("#64748b")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    title_s   = S("T",  fontSize=18, textColor=DARK, fontName="Helvetica-Bold", spaceAfter=3)
    sub_s     = S("Su", fontSize=9,  textColor=MID,  fontName="Helvetica",      spaceAfter=8)
    h2_s      = S("H2", fontSize=11, textColor=DARK, fontName="Helvetica-Bold",
                  spaceBefore=12, spaceAfter=5)
    body_s    = S("B",  fontSize=9,  textColor=DARK, fontName="Helvetica",
                  leading=13, spaceAfter=4)
    formula_s = S("Fm", fontSize=9,  textColor=DARK, fontName="Helvetica-Oblique",
                  leading=13, leftIndent=8)
    small_s   = S("Sm", fontSize=8,  textColor=MID,  fontName="Helvetica")

    def tbl(data, widths, hdr=True):
        t = Table(data, colWidths=widths)
        ts = [
            ("FONTNAME",     (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",     (0,0), (-1,-1), 8.5),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS",(0, 1 if hdr else 0), (-1,-1), [colors.white, BGREY]),
        ]
        if hdr:
            ts += [("BACKGROUND", (0,0),(-1,0), BLUE),
                   ("TEXTCOLOR",  (0,0),(-1,0), colors.white),
                   ("FONTNAME",   (0,0),(-1,0), "Helvetica-Bold")]
        else:
            ts += [("FONTNAME",   (0,0),(0,-1), "Helvetica-Bold")]
        t.setStyle(TableStyle(ts))
        return t

    story = []

    # ── Header ──────────────────────────────────────────────
    hdr_tbl = Table(
        [[Paragraph("RADCF Fair Pricing Report", title_s)],
         [Paragraph("Re-pricing Mobile Phones Sold on Hire Purchase · Egerton University, "
                    "Department of Mathematics · BSc Actuarial Science", sub_s)],
         [Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')} EAT  ·  "
                    f"ID: RADCF-{datetime.now().strftime('%Y%m%d-%H%M%S')}", small_s)]],
        colWidths=[17*cm])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), LBLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 12),
        ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("LINEBELOW",     (0,-1),(-1,-1), 2, BLUE),
    ]))
    story.extend([hdr_tbl, Spacer(1, 12)])

    # ── 1. Contract Summary ──────────────────────────────────
    story.append(Paragraph("1. Contract Summary (Inputs)", h2_s))
    story.append(tbl([
        ["Parameter", "Value"],
        ["Cash Price (OP)",             ksh(inputs["cash_price"])],
        ["Repayment Term",              f'{inputs["n_months"]} months'],
        ["Deposit",                     f'{inputs["deposit_pct"]:.1f}%  →  {ksh(radcf["deposit"])}'],
        ["Admin Cost",                  f'{inputs["admin_pct"]:.1f}%  →  {ksh(radcf["admin_cost"])}'],
        ["Monthly Discount Rate r",     f'{inputs["r_monthly"]:.4f}  ({inputs["r_monthly"]*100:.2f}%/month)'],
        ["Borrower Monthly Income",     ksh(inputs["income_ksh"])],
    ], [7*cm, 10*cm]))
    story.append(Spacer(1, 8))

    # ── 2. PD Model ──────────────────────────────────────────
    story.append(Paragraph("2. Probability of Default — Single-Variable Logistic Model", h2_s))
    story.append(Paragraph(
        "PD = 1 / (1 + exp(−(β₀ + β₁ · ln(Income / 1000))))", formula_s))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Fitted on KNBS KCHSP 2022 Kenya data (n=987 households after cleaning from 17,827 records). "
        "Income is the sole predictor — chosen for actuarial group-pricing purpose, data availability, "
        "and to avoid demographic discrimination risk.", small_s))
    story.append(Spacer(1, 5))
    lvl, _, pd_expl = pd_bucket(pd_value)
    z_val = BETA0 + BETA1 * math.log(max(1, inputs["income_ksh"]) / 1000.0)
    story.append(tbl([
        ["Parameter",           "Value",           "Note"],
        ["β₀ (intercept)",      f"{BETA0:.4f}",     "z = 4.807   |   p = 5.11×10⁻⁷"],
        ["β₁ (log income)",     f"{BETA1:.4f}",     "z = −6.963  |   p = 3.33×10⁻⁹   ·   higher income → lower PD"],
        ["ln(Income/1000)",     f'{math.log(max(1,inputs["income_ksh"])/1000):.4f}',
                                f'Income = {ksh(inputs["income_ksh"])}/month'],
        ["z  (linear combo)",   f"{z_val:.4f}",     "β₀ + β₁ · ln(I/1000)"],
        ["Estimated PD",        f'{pd_value:.4f}  ({pd_value*100:.1f}%)',
                                f'{lvl} Risk — {pd_expl}'],
    ], [5*cm, 3.5*cm, 8.5*cm]))
    story.append(Spacer(1, 8))

    # ── 3. RADCF Computation ─────────────────────────────────
    story.append(Paragraph("3. RADCF Pricing Computation", h2_s))
    for line in [
        f"CF_revised = OP + AdminCost − Deposit = {ksh(radcf['op'])} + "
        f"{ksh(radcf['admin_cost'])} − {ksh(radcf['deposit'])} = {ksh(radcf['cf_revised'])}",
        f"AF = (1 − (1+{inputs['r_monthly']:.4f})^(−{inputs['n_months']})) / "
        f"{inputs['r_monthly']:.4f}  =  {radcf['af']:.4f}",
        f"M = CF_revised / ((1−PD) × AF)  =  {ksh(radcf['cf_revised'])} / "
        f"({radcf['repay_prob']:.4f} × {radcf['af']:.4f})  =  {ksh(radcf['fair_monthly'])}",
    ]:
        story.append(Paragraph(line, formula_s))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 6))

    # ── 4. Fair Price Outputs ────────────────────────────────
    story.append(Paragraph("4. Fair Price Outputs (Main Results)", h2_s))
    story.append(tbl([
        ["Output", "Value"],
        ["Deposit",                         ksh(radcf["deposit"])],
        ["Fair Monthly Installment (M)",    ksh(radcf["fair_monthly"])],
        ["Fair Total Paid (Deposit + M×n)", ksh(radcf["fair_total"])],
        ["RADCF Present Value",             ksh(radcf["radcf_pv"])],
    ], [9*cm, 8*cm]))
    story.append(Spacer(1, 8))

    # ── 5. Market Comparison ─────────────────────────────────
    story.append(Paragraph("5. Market Comparison", h2_s))
    if market.get("provided"):
        tag, _, tag_expl = fairness_tag(market.get("over_pct", float("nan")))
        story.append(tbl([
            ["Metric", "Value"],
            ["Market Monthly Installment",  ksh(market.get("market_monthly", float("nan")))],
            ["Market Total Repayment",      ksh(market.get("market_total",   float("nan")))],
            ["RADCF Fair Total",            ksh(radcf["fair_total"])],
            ["Overpricing Amount",          ksh(market.get("over_amt",       float("nan")))],
            ["Overpricing (%)",             f'{market.get("over_pct",0)*100:.2f}%'],
            ["Fairness Score (0–100)",      f'{market.get("fairness_score",0):.1f}'],
            ["Assessment",                  f'{tag} — {tag_expl}'],
        ], [9*cm, 8*cm]))
    else:
        story.append(Paragraph("No market comparison values were provided.", body_s))
    story.append(Spacer(1, 8))

    # ── 6. Sensitivity ───────────────────────────────────────
    story.append(Paragraph("6. Sensitivity Analysis (Stress Test)", h2_s))
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        cols  = ["Scenario","PD","Admin %","r","Term (mo)","Fair Monthly (KSh)","Fair Total (KSh)"]
        tdata = [cols] + [list(r) for r in sensitivity_df[cols].itertuples(index=False)]
        t5    = Table(tdata, colWidths=[4*cm,1.5*cm,1.6*cm,1.4*cm,1.6*cm,3.2*cm,3.2*cm])
        t5.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), BLUE),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, BGREY]),
            ("ALIGN",         (1,1),(-1,-1), "RIGHT"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ]))
        story.append(t5)
    story.append(Spacer(1, 8))

    # ── 7. Conclusion ────────────────────────────────────────
    story.append(Paragraph("7. Conclusion & Recommendation", h2_s))
    lvl, _, _ = pd_bucket(pd_value)
    conclusion = (
        f"Based on the RADCF framework, the actuarially fair repayment plan for a borrower "
        f"earning {ksh(inputs['income_ksh'])}/month is: deposit {ksh(radcf['deposit'])}, "
        f"monthly installment {ksh(radcf['fair_monthly'])}, fair total {ksh(radcf['fair_total'])}. "
        f"The estimated probability of default is {pd_value*100:.1f}% ({lvl} risk). "
    )
    if market.get("provided"):
        tag, _, _ = fairness_tag(market.get("over_pct", float("nan")))
        conclusion += (f"The market deal is assessed as '{tag}' with "
                       f"{market.get('over_pct',0)*100:.2f}% overpricing vs RADCF fair value. ")
    conclusion += "Sensitivity results show which parameters most influence fair pricing."
    story.append(Paragraph(conclusion, body_s))
    story.append(Spacer(1, 8))

    # ── 8. Assumptions ───────────────────────────────────────
    story.append(Paragraph("8. Assumptions & Limitations", h2_s))
    for a in [
        "PD model uses a single predictor (monthly income). Fitted on KNBS KCHSP 2022 (n=987). "
        "β₀=1.5309 (p=5.11×10⁻⁷), β₁=−0.7322 (p=3.33×10⁻⁹).",
        "Income is entered as ln(Income/1000) to handle the right-skewed household income distribution.",
        "Deposit fixed at 30% and admin cost at 5% of cash price — standard Kenyan hire-purchase terms.",
        "PD is assumed constant over the repayment term (simplifying assumption).",
        "No recovery after default is assumed (Loss Given Default ≈ 100%).",
        "Market installment benchmark of KES 3,150/month sourced from M-Kopa and Watu Credit (2024–2025).",
        "The single-variable model is deliberately chosen for actuarial group-pricing — "
        "the goal is band-level default rate estimation, not individual credit scoring.",
    ]:
        story.append(Paragraph(f"• {a}", body_s))

    story.extend([
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")),
        Spacer(1, 6),
        Paragraph(
            "RADCF Fair Pricing Engine · Egerton University, Department of Mathematics · "
            "BSc Actuarial Science Group Project", small_s),
    ])
    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RADCF Fair Pricing Engine",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed",
)

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
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:var(--bg); }
#MainMenu, footer, header  { visibility:hidden; }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1360px; }

/* ── Hero ── */
.hero {
  background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#1d4ed8 100%);
  border-radius:20px; padding:2.8rem 2.5rem 2rem;
  margin-bottom:2rem; position:relative; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; top:-70px; right:-70px;
  width:320px; height:320px;
  background:radial-gradient(circle,rgba(59,130,246,.22) 0%,transparent 70%);
  border-radius:50%;
}
.hero-label { font-size:.7rem; font-weight:700; letter-spacing:.18em;
              text-transform:uppercase; color:#93c5fd; margin-bottom:.5rem; }
.hero-title { font-family:'Playfair Display',serif; font-size:clamp(1.6rem,3vw,2.3rem);
              font-weight:700; color:#fff; line-height:1.25; margin-bottom:.7rem; }
.hero-sub   { font-size:.9rem; color:#93c5fd; max-width:660px; line-height:1.65; margin-bottom:1.4rem; }
.hero-badges { display:flex; gap:.55rem; flex-wrap:wrap; }
.badge { display:inline-flex; align-items:center; gap:.3rem; font-size:.73rem; font-weight:500;
         padding:.28rem .75rem; background:rgba(255,255,255,.12);
         border:1px solid rgba(255,255,255,.2); border-radius:100px; color:#e0f2fe; }

/* ── Section labels ── */
.sec-label { font-size:.67rem; font-weight:700; letter-spacing:.14em;
             text-transform:uppercase; color:var(--lblue); margin-bottom:.2rem; }
.sec-title { font-family:'Playfair Display',serif; font-size:1.28rem; font-weight:600;
             color:var(--navy); margin-bottom:.9rem; }

/* ── Metric cards ── */
.mcard {
  background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:1.15rem 1.25rem; position:relative; overflow:hidden;
  transition:transform .18s,box-shadow .18s;
}
.mcard:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,.08); }
.mcard .bar  { position:absolute; top:0; left:0; width:100%; height:3px; }
.mlabel { font-size:.71rem; font-weight:700; color:var(--slate);
          letter-spacing:.07em; text-transform:uppercase; margin-bottom:.28rem; }
.mvalue { font-family:'DM Mono',monospace; font-size:1.4rem;
          font-weight:500; color:var(--navy); line-height:1.2; }
.msub   { font-size:.74rem; color:var(--slate); margin-top:.25rem; }

/* ── Formula box ── */
.fbox {
  background:#0f172a; border-radius:12px; padding:1.15rem 1.4rem;
  font-family:'DM Mono',monospace; font-size:.82rem; color:#93c5fd;
  line-height:2.1; margin:.65rem 0; white-space:pre;
}
.fbox .var { color:#fbbf24; }
.fbox .num { color:#34d399; }
.fbox .res { color:#ffffff; font-weight:600; }
.fbox .dim { color:#475569; }

/* ── Info box ── */
.ibox { background:var(--ice); border:1px solid #bfdbfe; border-radius:10px;
        padding:.78rem 1rem; font-size:.82rem; color:#1e40af; margin:.4rem 0; line-height:1.6; }

/* ── Affordability bar ── */
.iti-wrap { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
            padding:.9rem 1.1rem; margin-top:.5rem; }
.iti-track { background:#e2e8f0; border-radius:100px; height:8px;
             overflow:hidden; margin:.4rem 0 .3rem; }

/* ── Comparison bars ── */
.cbar-wrap { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
             padding:.95rem 1.2rem; margin-top:.5rem; }
.crow  { display:flex; align-items:center; gap:.7rem; margin-bottom:.5rem; }
.cname { font-size:.77rem; font-weight:600; width:130px; color:var(--navy); flex-shrink:0; }
.ctrack { flex:1; background:var(--border); border-radius:100px; height:9px; overflow:hidden; }
.cfill  { height:100%; border-radius:100px; }
.cval   { font-family:'DM Mono',monospace; font-size:.76rem;
          color:var(--slate); min-width:95px; text-align:right; }

/* ── Overpricing banner ── */
.banner { border-radius:13px; padding:1.05rem 1.35rem; margin:.65rem 0; border-left:4px solid; }
.b-severe { background:#fef2f2; border-color:#ef4444; }
.b-over   { background:#fffbeb; border-color:#f59e0b; }
.b-fair   { background:#f0fdf4; border-color:#22c55e; }
.b-below  { background:#eff6ff; border-color:#3b82f6; }
.b-tag  { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
.b-desc { font-size:.81rem; color:#334155; margin-top:.2rem; }

/* ── Band chip ── */
.band-chip { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
             padding:.75rem 1rem; margin-top:.5rem; }
.band-label { font-size:.67rem; font-weight:700; text-transform:uppercase;
              letter-spacing:.08em; color:#166534; margin-bottom:.2rem; }
.band-name  { font-size:.88rem; font-weight:600; color:#14532d; }
.band-dr    { font-size:.76rem; color:#15803d; margin-top:.2rem; }

/* ── Steps ── */
.steps { display:flex; align-items:flex-start; gap:0; margin-bottom:1.8rem; }
.step  { flex:1; text-align:center; }
.scircle { width:33px; height:33px; border-radius:50%; background:var(--blue);
           color:white; font-weight:700; font-size:.87rem;
           display:inline-flex; align-items:center; justify-content:center;
           margin-bottom:.3rem; }
.stext  { font-size:.67rem; font-weight:600; color:var(--slate);
          letter-spacing:.05em; text-transform:uppercase; }
.sline  { flex:1; height:2px; background:var(--border); margin-top:16px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:var(--card); border-radius:12px; border:1px solid var(--border); padding:4px; gap:2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius:9px; padding:.44rem 1.1rem; font-weight:500; font-size:.86rem; color:var(--slate);
}
.stTabs [aria-selected="true"] { background:var(--blue) !important; color:white !important; }

/* ── Inputs ── */
.stNumberInput label { font-size:.81rem !important; font-weight:600 !important; color:var(--navy) !important; }
.stNumberInput > div > div > input {
  border-radius:8px !important; font-family:'DM Mono',monospace !important; font-size:.87rem !important;
}

/* ── Download button ── */
.stDownloadButton button {
  background:linear-gradient(135deg,var(--blue) 0%,#1e40af 100%) !important;
  color:white !important; border:none !important; border-radius:10px !important;
  padding:.6rem 1.5rem !important; font-weight:600 !important;
  box-shadow:0 4px 14px rgba(29,78,216,.3) !important;
}
.stDownloadButton button:hover { transform:translateY(-1px) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  font-weight:600 !important; font-size:.84rem !important;
  color:var(--navy) !important; background:var(--bg) !important; border-radius:8px !important;
}
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-label">Actuarial Research Tool · Egerton University · BSc Actuarial Science</div>
  <div class="hero-title">RADCF Fair Pricing Engine</div>
  <div class="hero-sub">
    Actuarial re-pricing of mobile phones sold on hire purchase in Kenya.
    Uses a Risk-Adjusted Discounted Cash Flow model with a single-variable logistic
    default estimator calibrated on KNBS KCHSP 2022 data (n=987 households).
  </div>
  <div class="hero-badges">
    <span class="badge">📊 Single-Variable Logistic PD Model</span>
    <span class="badge">📦 KNBS KCHSP 2022 · n=987</span>
    <span class="badge">🧮 β₀=1.5309 · β₁=−0.7322</span>
    <span class="badge">🏛 Risk-Adjusted DCF Pricing</span>
    <span class="badge">📄 PDF Report Export</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Model Overview ──────────────────────────────────────────────────────────
with st.expander("📖 Model Overview & Mathematical Framework", expanded=False):
    ov1, ov2 = st.columns([1.2, 0.8])
    with ov1:
        st.markdown("""
**Research Context**

This tool implements the RADCF pricing model from the Egerton University actuarial
science group project. Kenya's hire-purchase market charges a flat repayment rate
regardless of borrower income or credit risk, resulting in systematic overpricing —
particularly for higher-income borrowers.

**Why income as the only predictor?**
- The KCHSP 2022 dataset lacks borrower-level credit history
- The model prices *risk pools* (income bands), not individuals — income is sufficient for this
- Additional variables (age, gender, region) introduce demographic discrimination risk
- Income is verifiable via M-PESA transaction history at the point of contract

**Workflow**
1. Estimate **PD** from monthly income via logistic regression
2. Adjust repayments using **(1 − PD)**
3. Discount cash flows at monthly rate **r** via the annuity factor **AF**
4. Add deposit and admin costs → **fair monthly installment M**
5. Compare against market price → quantify overpricing
        """)
        coef_df = pd.DataFrame({
            "Coefficient": ["β₀ (intercept)", "β₁ (log income)"],
            "Value":       [BETA0, BETA1],
            "z-statistic": ["4.807", "−6.963"],
            "p-value":     ["5.11×10⁻⁷", "3.33×10⁻⁹"],
            "Interpretation": ["Baseline log-odds", "Higher income → lower PD"],
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
        st.caption("AUC reference: model validated against raw default rates (42.3% → 14.7% across bands).")

    with ov2:
        st.markdown("**Core Formulas**")
        st.latex(r"PD = \frac{1}{1+e^{-(\beta_0 + \beta_1 \ln(I/1000))}}")
        st.latex(r"AF = \frac{1-(1+r)^{-n}}{r}")
        st.latex(r"CF_{rev} = OP + \text{Admin} - \text{Deposit}")
        st.latex(r"M = \frac{CF_{rev}}{(1-PD)\cdot AF}")
        st.caption("Where I = monthly income (KSh), r = monthly discount rate, n = term (months)")
        st.markdown("**Raw Default Rates by Income Band (KCHSP 2022)**")
        bdf = pd.DataFrame([
            {"Income Band": b, "Midpoint": f"KSh {m:,}", "Raw Default Rate": f"{d*100:.1f}%"}
            for b, m, d in INCOME_BANDS
        ])
        st.dataframe(bdf, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🧮  Manual Calculator", "📋  Paste Contract Text"])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — MANUAL CALCULATOR
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="steps">
      <div class="step"><div class="scircle">1</div><div class="stext">Contract</div></div>
      <div class="sline"></div>
      <div class="step"><div class="scircle">2</div><div class="stext">Income</div></div>
      <div class="sline"></div>
      <div class="step"><div class="scircle">3</div><div class="stext">Results</div></div>
      <div class="sline"></div>
      <div class="step"><div class="scircle">4</div><div class="stext">Market</div></div>
      <div class="sline"></div>
      <div class="step"><div class="scircle">5</div><div class="stext">Report</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUTS ────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Steps 1 & 2</div>'
                '<div class="sec-title">Contract & Borrower Inputs</div>',
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown("**📄 Contract Details**")
        cash_price = st.number_input(
            "Cash price (KSh)", min_value=0.0, value=20000.0, step=500.0, key="cp1",
            help="Retail cash price of the smartphone. Default KSh 20,000 = paper reference price.")
        n_months = st.number_input(
            "Repayment term (months)", min_value=1, value=12, step=1, key="n1",
            help="Standard term in the Kenyan market is 12 months.")
        deposit_pct = st.number_input(
            "Deposit (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key="dp1",
            help="Standard market deposit: 30% of cash price (per M-Kopa / Watu Credit contracts).")
        admin_cost_pct = st.number_input(
            "Administrative cost (%)", min_value=0.0, max_value=30.0, value=5.0,
            step=0.5, key="ad1",
            help="Processing/admin fee as % of cash price. Standard market rate: 5%.")

    with col_b:
        st.markdown("**📈 Discount Rate**")
        r_monthly = st.number_input(
            "Monthly discount rate r",
            min_value=0.0, value=round(0.02, 4),
            step=0.001, format="%.4f", key="r1",
            help="Paper uses 2%/month (24% p.a.). CBK rate 13% p.a. → 0.0108/month.")

        af_display = annuity_factor(r_monthly, int(n_months))
        st.markdown(f"""
        <div class="ibox">
        💡 <strong>r = {r_monthly:.4f}/month</strong><br>
        Effective APR ≈ <strong>{((1+r_monthly)**12-1)*100:.1f}% p.a.</strong><br>
        Annuity factor (n = {int(n_months)} mo) = <strong>{af_display:.4f}</strong>
        </div>""", unsafe_allow_html=True)

        st.markdown("**⚙️ Advanced: Override Coefficients**")
        with st.expander("Override β₀ and β₁ (optional)", expanded=False):
            beta0_in = st.number_input(
                "β₀ (intercept)", value=BETA0, step=0.0001, format="%.4f", key="b0_in")
            beta1_in = st.number_input(
                "β₁ (log income)", value=BETA1, step=0.0001, format="%.4f", key="b1_in")

        # Read from session state — always valid
        beta0_use = float(st.session_state.get("b0_in", BETA0))
        beta1_use = float(st.session_state.get("b1_in", BETA1))

    with col_c:
        st.markdown("**👤 Borrower Monthly Income**")
        income_ksh = st.number_input(
            "Monthly income (KSh)", min_value=0.0, value=20000.0, step=1000.0, key="inc1",
            help="The borrower's verified monthly income. This is the sole predictor in the PD model.")

        # Identify income band
        band_name = "—"
        band_dr   = None
        for bname, mid_inc, raw_dr in INCOME_BANDS:
            parts = bname.replace("KES ", "").replace(",", "")
            if "Below" in bname:
                lo, hi = 0, 10000
            elif "Above" in bname:
                lo, hi = 70000, float("inf")
            else:
                lo_s, hi_s = parts.split("–")
                lo, hi = float(lo_s.strip()), float(hi_s.strip())
            if lo <= income_ksh <= hi:
                band_name = bname
                band_dr   = raw_dr
                break

        dr_html = (f'<div class="band-dr">Raw default rate (KCHSP 2022): '
                   f'<strong>{band_dr*100:.1f}%</strong></div>' if band_dr else "")
        st.markdown(f"""
        <div class="band-chip">
          <div class="band-label">Income Band</div>
          <div class="band-name">{band_name}</div>
          {dr_html}
        </div>""", unsafe_allow_html=True)

    # ── COMPUTE ────────────────────────────────────────────────────
    pd_val = logistic_pd(income_ksh, beta0=beta0_use, beta1=beta1_use)
    res    = fair_installment(cash_price, deposit_pct, admin_cost_pct,
                              int(n_months), float(r_monthly), pd_val)
    lvl, lvl_color, lvl_expl = pd_bucket(pd_val)
    z_val  = beta0_use + beta1_use * math.log(max(1, income_ksh) / 1000.0)

    st.divider()

    # ── RESULTS ────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Step 3</div>'
                '<div class="sec-title">RADCF Fair Pricing Results</div>',
                unsafe_allow_html=True)

    gauge_col, metrics_col = st.columns([0.37, 0.63], gap="large")

    with gauge_col:
        # PD Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pd_val * 100, 1),
            number={"suffix": "%", "font": {"size": 38, "family": "DM Mono", "color": "#0f172a"}},
            title={"text": "Probability of Default", "font": {"size": 12, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8",
                         "tickvals": [0, 25, 50, 75, 100]},
                "bar":  {"color": lvl_color, "thickness": 0.28},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0,  25],  "color": "#dcfce7"},
                    {"range": [25, 50],  "color": "#fef9c3"},
                    {"range": [50, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 2},
                              "thickness": 0.75, "value": pd_val * 100},
            }
        ))
        fig_g.update_layout(
            height=235, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="background:{lvl_color}18;border:1px solid {lvl_color}44;
             border-radius:10px;padding:.72rem 1rem;text-align:center">
          <div style="font-size:.7rem;font-weight:700;letter-spacing:.1em;
               text-transform:uppercase;color:{lvl_color}">{lvl} Risk</div>
          <div style="font-size:.77rem;color:#334155;line-height:1.5;
               margin-top:.2rem">{lvl_expl}</div>
        </div>""", unsafe_allow_html=True)

        # PD breakdown
        with st.expander("PD computation breakdown"):
            inc_c = beta1_use * math.log(max(income_ksh, 1) / 1000.0)
            st.markdown(f"""<div class="fbox"><span class="var">β₀</span> (baseline)          <span class="num">{beta0_use:+.4f}</span>
<span class="var">β₁</span> × ln({income_ksh:,.0f}/1000)  <span class="num">{inc_c:+.4f}</span>
<span class="dim">─────────────────────────</span>
<span class="var">z</span>  (linear combo)       <span class="res">{z_val:+.4f}</span>
<span class="var">PD</span> = 1/(1+e^(−z))      <span class="res">{pd_val:.4f}</span></div>""",
                        unsafe_allow_html=True)

    with metrics_col:
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"""<div class="mcard">
          <div class="bar" style="background:linear-gradient(90deg,#1d4ed8,#3b82f6)"></div>
          <div class="mlabel">Fair Monthly</div>
          <div class="mvalue">KSh {res['fair_monthly']:,.0f}</div>
          <div class="msub">per month × {n_months} mo</div>
        </div>""", unsafe_allow_html=True)
        mc2.markdown(f"""<div class="mcard">
          <div class="bar" style="background:linear-gradient(90deg,#0891b2,#06b6d4)"></div>
          <div class="mlabel">Deposit</div>
          <div class="mvalue">KSh {res['deposit']:,.0f}</div>
          <div class="msub">{deposit_pct:.0f}% upfront</div>
        </div>""", unsafe_allow_html=True)
        mc3.markdown(f"""<div class="mcard">
          <div class="bar" style="background:linear-gradient(90deg,#7c3aed,#a78bfa)"></div>
          <div class="mlabel">Admin Cost</div>
          <div class="mvalue">KSh {res['admin_cost']:,.0f}</div>
          <div class="msub">{admin_cost_pct:.0f}% of cash price</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        mc4, mc5 = st.columns(2)
        markup = (res['fair_total'] - cash_price) / cash_price * 100 if cash_price > 0 else 0
        mc4.markdown(f"""<div class="mcard">
          <div class="bar" style="background:linear-gradient(90deg,#059669,#34d399)"></div>
          <div class="mlabel">Fair Total Paid</div>
          <div class="mvalue">KSh {res['fair_total']:,.0f}</div>
          <div class="msub">+{markup:.1f}% above cash price</div>
        </div>""", unsafe_allow_html=True)
        mc5.markdown(f"""<div class="mcard">
          <div class="bar" style="background:linear-gradient(90deg,#dc2626,#f87171)"></div>
          <div class="mlabel">RADCF Present Value</div>
          <div class="mvalue">KSh {res['radcf_pv']:,.0f}</div>
          <div class="msub">Expected PV of repayments</div>
        </div>""", unsafe_allow_html=True)

        # RADCF formula trace
        st.markdown(f"""<div class="fbox"><span class="var">CF_rev</span> = {ksh(cash_price)} + {ksh(res['admin_cost'])} − {ksh(res['deposit'])} = <span class="res">{ksh(res['cf_revised'])}</span>
<span class="var">AF</span>     = (1−(1+{r_monthly:.4f})^(−{n_months})) / {r_monthly:.4f} = <span class="res">{res['af']:.4f}</span>
<span class="var">M</span>      = {ksh(res['cf_revised'])} / ({res['repay_prob']:.4f} × {res['af']:.4f}) = <span class="res">{ksh(res['fair_monthly'])}</span></div>""",
                    unsafe_allow_html=True)

        # Affordability ITI
        if income_ksh > 0:
            iti   = res['fair_monthly'] / income_ksh * 100
            ic    = "#ef4444" if iti > 40 else "#f59e0b" if iti > 30 else "#22c55e"
            label = ("High burden (>40%)" if iti > 40
                     else "Moderate (30–40%)" if iti > 30
                     else "Affordable (<30%)")
            st.markdown(f"""
            <div class="iti-wrap">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:.75rem;font-weight:700;color:#334155;
                     text-transform:uppercase;letter-spacing:.06em">Affordability · ITI Ratio</span>
                <span style="font-family:'DM Mono',monospace;font-size:1rem;
                     font-weight:600;color:{ic}">{iti:.1f}%</span>
              </div>
              <div class="iti-track">
                <div style="background:{ic};width:{min(100,iti):.1f}%;
                     height:100%;border-radius:100px"></div>
              </div>
              <div style="font-size:.74rem;color:{ic};font-weight:600">{label}</div>
              <div style="font-size:.71rem;color:#64748b;margin-top:.15rem">
                Fair installment as % of KSh {income_ksh:,.0f}/month · 30% sustainability threshold
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── INCOME BAND TABLE ─────────────────────────────────────────
    with st.expander("📊 Full Income Band Comparison — all 6 bands (KSh 20,000 phone, paper defaults)", expanded=False):
        band_rows = []
        for bname, mid_inc, raw_dr in INCOME_BANDS:
            pd_b  = logistic_pd(mid_inc)
            r_b   = fair_installment(20000.0, 30.0, 5.0, 12, 0.02, pd_b)
            over  = (MARKET_INSTALLMENT - r_b["fair_monthly"]) / r_b["fair_monthly"] * 100
            itib  = r_b["fair_monthly"] / mid_inc * 100
            band_rows.append({
                "Income Band":         bname,
                "Midpoint (KSh)":      f"{mid_inc:,}",
                "PD — Model (%)":      f"{pd_b*100:.1f}",
                "PD — Raw data (%)":   f"{raw_dr*100:.1f}",
                "Fair Monthly (KSh)":  f"{r_b['fair_monthly']:,.0f}",
                "Market KSh 3,150":    f"{MARKET_INSTALLMENT:,.0f}",
                "Overcharge (%)":      f"+{over:.1f}",
                "ITI (fair price)":    f"{itib:.1f}%",
            })
        st.dataframe(pd.DataFrame(band_rows), hide_index=True, use_container_width=True)
        st.caption("Market benchmark: KSh 3,150/month (M-Kopa / Watu Credit 2024–2025 · r=2%/month)")

    # ── MARKET COMPARISON ─────────────────────────────────────────
    st.markdown('<div class="sec-label">Step 4</div>'
                '<div class="sec-title">Market Deal Comparison</div>',
                unsafe_allow_html=True)

    mkt_l, mkt_r = st.columns([0.44, 0.56], gap="large")
    market_info  = {"provided": False}

    with mkt_l:
        st.markdown("**Enter the actual market contract pricing to assess overpricing:**")
        market_monthly  = st.number_input(
            "Market monthly installment (KSh)", min_value=0.0,
            value=0.0, step=100.0, key="mkt_mo",
            help="Actual installment charged (e.g. M-Kopa KSh 3,150).")
        market_total_in = st.number_input(
            "Market total repayment (KSh)", min_value=0.0,
            value=0.0, step=500.0, key="mkt_tot",
            help="Total paid over full term. Overrides monthly if entered.")
        st.caption("💡 Leave both at 0 to skip market comparison.")

        # Reference display cards (static — no session state interaction)
        st.markdown("**Common market references:**")
        r1c, r2c, r3c = st.columns(3)
        r1c.markdown("""<div style="background:#fef2f2;border:1px solid #fecaca;
            border-radius:8px;padding:.6rem;text-align:center;font-size:.77rem">
            <strong>M-Kopa</strong><br><span style="font-family:'DM Mono',monospace">
            KSh 3,150</span></div>""", unsafe_allow_html=True)
        r2c.markdown("""<div style="background:#fef2f2;border:1px solid #fecaca;
            border-radius:8px;padding:.6rem;text-align:center;font-size:.77rem">
            <strong>Watu Credit</strong><br><span style="font-family:'DM Mono',monospace">
            KSh 3,500</span></div>""", unsafe_allow_html=True)
        r3c.markdown("""<div style="background:#fffbeb;border:1px solid #fde68a;
            border-radius:8px;padding:.6rem;text-align:center;font-size:.77rem">
            <strong>Market Avg</strong><br>~89% markup</div>""", unsafe_allow_html=True)

    with mkt_r:
        fair_total     = res["fair_total"]
        over_amt       = float("nan")
        over_pct       = float("nan")
        implied_apr    = float("nan")
        mkt_total_used = 0.0

        if market_total_in > 0:
            mkt_total_used = float(market_total_in)
            over_amt = mkt_total_used - fair_total
            over_pct = (over_amt / fair_total) if fair_total > 0 else float("nan")
        elif market_monthly > 0:
            mkt_total_used = res["deposit"] + float(market_monthly) * int(n_months)
            over_amt       = mkt_total_used - fair_total
            over_pct       = (over_amt / fair_total) if fair_total > 0 else float("nan")
            principal      = cash_price - res["deposit"]
            im             = implied_monthly_rate(principal, float(market_monthly), int(n_months))
            implied_apr    = effective_apr(im)

        if np.isfinite(over_pct):
            tag, tag_col, tag_expl = fairness_tag(over_pct)
            fs   = max(0.0, min(100.0, 100.0 - over_pct * 100.0))
            bcls = ("b-severe" if over_pct >= 0.25 else
                    "b-over"   if over_pct >= 0.10 else
                    "b-fair"   if over_pct >= -0.10 else "b-below")

            st.markdown(f"""
            <div class="banner {bcls}">
              <div class="b-tag" style="color:{tag_col}">{tag}</div>
              <div class="b-desc">{tag_expl}</div>
            </div>""", unsafe_allow_html=True)

            oc1, oc2, oc3 = st.columns(3)
            oc1.markdown(f"""<div class="mcard">
              <div class="bar" style="background:{tag_col}"></div>
              <div class="mlabel">Overpricing</div>
              <div class="mvalue" style="color:{tag_col}">{over_pct*100:.1f}%</div>
              <div class="msub">vs RADCF fair value</div>
            </div>""", unsafe_allow_html=True)
            oc2.markdown(f"""<div class="mcard">
              <div class="bar" style="background:#ef4444"></div>
              <div class="mlabel">Extra Paid</div>
              <div class="mvalue" style="font-size:1.1rem">KSh {over_amt:,.0f}</div>
              <div class="msub">above fair total</div>
            </div>""", unsafe_allow_html=True)
            oc3.markdown(f"""<div class="mcard">
              <div class="bar" style="background:#7c3aed"></div>
              <div class="mlabel">Fairness Score</div>
              <div class="mvalue">{fs:.0f}<span style="font-size:.9rem">/100</span></div>
              <div class="msub">0 = worst · 100 = fair</div>
            </div>""", unsafe_allow_html=True)

            # Visual bar comparison
            max_v = max(fair_total, mkt_total_used, cash_price, 1)
            st.markdown(f"""
            <div class="cbar-wrap">
              <div style="font-size:.72rem;font-weight:700;color:#334155;
                   text-transform:uppercase;letter-spacing:.06em;margin-bottom:.6rem">
                Total Repayment Comparison</div>
              <div class="crow">
                <div class="cname">RADCF Fair</div>
                <div class="ctrack"><div class="cfill"
                  style="width:{fair_total/max_v*100:.1f}%;background:#22c55e"></div></div>
                <div class="cval">KSh {fair_total:,.0f}</div>
              </div>
              <div class="crow">
                <div class="cname">Market Price</div>
                <div class="ctrack"><div class="cfill"
                  style="width:{mkt_total_used/max_v*100:.1f}%;background:{tag_col}"></div></div>
                <div class="cval">KSh {mkt_total_used:,.0f}</div>
              </div>
              <div class="crow">
                <div class="cname">Cash Price</div>
                <div class="ctrack"><div class="cfill"
                  style="width:{cash_price/max_v*100:.1f}%;background:#94a3b8"></div></div>
                <div class="cval">KSh {cash_price:,.0f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if np.isfinite(implied_apr):
                st.markdown(f"""
                <div class="ibox" style="margin-top:.5rem">
                📈 <strong>Implied effective APR</strong> on market deal:
                <strong>{implied_apr*100:.1f}% p.a.</strong>
                &nbsp;·&nbsp; vs discount rate used in model: {r_monthly*12*100:.1f}% p.a.
                </div>""", unsafe_allow_html=True)

            market_info = {
                "provided":       True,
                "market_monthly": float(market_monthly) if market_monthly > 0 else float("nan"),
                "market_total":   float(mkt_total_used),
                "over_amt":       float(over_amt),
                "over_pct":       float(over_pct),
                "fairness_score": float(fs),
                "tag":            tag,
                "tag_explain":    tag_expl,
                "implied_apr":    float(implied_apr),
            }
        else:
            st.markdown("""
            <div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
                 padding:2rem;text-align:center;color:#94a3b8">
              <div style="font-size:2rem;margin-bottom:.5rem">📊</div>
              <div style="font-size:.87rem;font-weight:500">
                Enter a market installment or total repayment on the left to compare
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── SENSITIVITY ANALYSIS ──────────────────────────────────────
    st.markdown('<div class="sec-label">Step 5</div>'
                '<div class="sec-title">Sensitivity Analysis & Stress Testing</div>',
                unsafe_allow_html=True)

    sens_df = run_scenarios(cash_price, deposit_pct, n_months, r_monthly,
                            pd_val, admin_cost_pct)

    sa_l, sa_r = st.columns([0.48, 0.52], gap="large")
    with sa_l:
        st.markdown("**Scenario Table**")
        st.dataframe(
            sens_df.style.format({
                "PD":                 "{:.3f}",
                "Admin %":            "{:.1f}",
                "r":                  "{:.4f}",
                "Fair Monthly (KSh)": "{:,.0f}",
                "Fair Total (KSh)":   "{:,.0f}",
            }).background_gradient(
                subset=["Fair Monthly (KSh)", "Fair Total (KSh)"], cmap="Blues"),
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
            bar_colors = ["#3b82f6" if v < 0 else "#ef4444"
                          for v in plot_df["Impact"]]
            fig_t = go.Figure(go.Bar(
                y=plot_df["Scenario"], x=plot_df["Impact"], orientation="h",
                marker_color=bar_colors,
                text=[f"KSh {v:+,.0f}" for v in plot_df["Impact"]],
                textposition="outside",
                textfont=dict(size=10, family="DM Mono"),
            ))
            fig_t.add_vline(x=0, line_width=1.5, line_color="#0f172a")
            fig_t.update_layout(
                height=370,
                xaxis_title=f"Δ {metric} vs Base (KSh {base_val:,.0f})",
                yaxis_title=None,
                title=dict(text="Tornado Chart — Sensitivity Impact",
                           font=dict(size=13, family="DM Sans"), x=0),
                margin=dict(l=10, r=90, t=45, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                yaxis=dict(tickfont=dict(size=11, family="DM Sans")),
                xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#94a3b8"),
            )
            st.plotly_chart(fig_t, use_container_width=True,
                            config={"displayModeBar": False})
            most = plot_df.iloc[-1]
            st.caption(f"Most sensitive factor: **{most['Scenario']}** "
                       f"→ KSh {most['Impact']:+,.0f} vs base")

    st.divider()

    # ── PDF DOWNLOAD ───────────────────────────────────────────────
    st.markdown("### 📄 Download Full Report")
    dl_c, info_c = st.columns([0.3, 0.7])
    with dl_c:
        pdf_bytes = build_pdf(
            inputs={
                "cash_price":   cash_price,
                "n_months":     int(n_months),
                "deposit_pct":  float(deposit_pct),
                "admin_pct":    float(admin_cost_pct),
                "r_monthly":    float(r_monthly),
                "income_ksh":   float(income_ksh),
            },
            pd_value=float(pd_val),
            radcf=res,
            market=market_info,
            sensitivity_df=sens_df,
        )
        st.download_button(
            label="⬇  Download RADCF Report (PDF)",
            data=pdf_bytes,
            file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="dl_pdf_manual",
            use_container_width=True,
        )
    with info_c:
        st.markdown("""
        <div class="ibox">
        The PDF report includes: contract summary, PD model parameters and computation,
        full RADCF pricing derivation, fair price outputs, market comparison (if entered),
        sensitivity analysis table, conclusion, and assumptions/limitations.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — PASTE CONTRACT TEXT
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-label">Smart Extraction</div>'
                '<div class="sec-title">Paste Contract or Offer Text</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="ibox">
    📋 Paste a hire-purchase offer — WhatsApp message, SMS, advert text, or contract snippet.
    The engine extracts key fields using pattern matching and computes the RADCF fair price.
    </div>""", unsafe_allow_html=True)

    sample = ("Cash price: KSh 20000. Deposit 30%. "
              "Pay KES 3150 per month for 12 months. Admin fee 5%.")
    txt = st.text_area(
        "Paste contract or offer text here", value=sample, height=120, key="txt2",
        placeholder="e.g. Cash price: KSh 18000. Deposit 30%. Monthly KES 3150 for 12 months. Admin 5%.")

    extracted = extract_deal_fields(txt)

    # Extraction result cards
    st.markdown("**Extracted fields:**")
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    def det_card(col, icon, label, val):
        found = val is not None
        col.markdown(f"""
        <div style="background:{'#f0fdf4' if found else '#fef2f2'};
             border:1px solid {'#bbf7d0' if found else '#fecaca'};
             border-radius:10px;padding:.7rem;text-align:center">
          <div style="font-size:1.2rem">{icon}</div>
          <div style="font-size:.67rem;font-weight:700;color:#64748b;
               text-transform:uppercase;letter-spacing:.07em">{label}</div>
          <div style="font-size:.84rem;font-weight:600;margin-top:.18rem;
               color:{'#166534' if found else '#991b1b'}">
               {val if val is not None else '—'}</div>
        </div>""", unsafe_allow_html=True)

    det_card(ec1, "💰", "Cash Price",
             f"KSh {extracted['cash_price']:,.0f}" if extracted["cash_price"] else None)
    dep_disp = (f"{extracted['deposit_pct']}%" if extracted["deposit_pct"]
                else (f"KSh {extracted['deposit_amount']:,.0f}"
                      if extracted["deposit_amount"] else None))
    det_card(ec2, "🏦", "Deposit",    dep_disp)
    det_card(ec3, "📅", "Term",
             f"{extracted['term_months']} months" if extracted["term_months"] else None)
    det_card(ec4, "💳", "Monthly",
             f"KSh {extracted['monthly_installment']:,.0f}"
             if extracted["monthly_installment"] else None)
    adm_disp = (f"{extracted['admin_pct']}%" if extracted["admin_pct"]
                else (f"KSh {extracted['admin_amount']:,.0f}"
                      if extracted["admin_amount"] else None))
    det_card(ec5, "⚙️", "Admin Fee", adm_disp)

    st.markdown("---")
    st.markdown("**Adjust extracted values if needed, then enter borrower income:**")

    ax, ay, az = st.columns(3, gap="large")
    with ax:
        cp2 = st.number_input("Cash price (KSh)", min_value=0.0,
                              value=float(extracted["cash_price"] or 20000.0),
                              step=500.0, key="cp2")
        nm2 = st.number_input("Repayment term (months)", min_value=1,
                              value=int(extracted["term_months"] or 12),
                              step=1, key="n2")
    with ay:
        dep_g = extracted["deposit_pct"]
        if dep_g is None and extracted["deposit_amount"] and cp2 > 0:
            dep_g = 100.0 * float(extracted["deposit_amount"]) / float(cp2)
        dp2 = st.number_input("Deposit (%)", min_value=0.0, max_value=100.0,
                              value=float(dep_g or 30.0), step=1.0, key="dp2")
        adm_g = extracted["admin_pct"]
        if adm_g is None and extracted["admin_amount"] and cp2 > 0:
            adm_g = 100.0 * float(extracted["admin_amount"]) / float(cp2)
        ad2 = st.number_input("Admin cost (%)", min_value=0.0, max_value=30.0,
                              value=float(adm_g or 5.0), step=0.5, key="ad2")
        r2  = st.number_input("Monthly discount rate r", min_value=0.0,
                              value=round(0.02, 4), step=0.001, format="%.4f", key="r2")
    with az:
        inc2 = st.number_input("Borrower monthly income (KSh)", min_value=0.0,
                               value=20000.0, step=1000.0, key="inc2",
                               help="The borrower's monthly income — sole predictor in the PD model.")

    pd2  = logistic_pd(inc2)
    res2 = fair_installment(cp2, dp2, ad2, int(nm2), float(r2), pd2)
    lvl2, lc2, _ = pd_bucket(pd2)

    st.divider()
    ra, rb, rc, rd = st.columns(4)
    ra.markdown(f"""<div class="mcard">
      <div class="bar" style="background:{lc2}"></div>
      <div class="mlabel">Probability of Default</div>
      <div class="mvalue">{pd2*100:.1f}%</div>
      <div class="msub">{lvl2} risk</div>
    </div>""", unsafe_allow_html=True)
    rb.markdown(f"""<div class="mcard">
      <div class="bar" style="background:#1d4ed8"></div>
      <div class="mlabel">Fair Monthly</div>
      <div class="mvalue">KSh {res2['fair_monthly']:,.0f}</div>
      <div class="msub">per month</div>
    </div>""", unsafe_allow_html=True)
    rc.markdown(f"""<div class="mcard">
      <div class="bar" style="background:#059669"></div>
      <div class="mlabel">Fair Total Paid</div>
      <div class="mvalue">KSh {res2['fair_total']:,.0f}</div>
      <div class="msub">deposit + installments</div>
    </div>""", unsafe_allow_html=True)

    mkt_inst2 = extracted.get("monthly_installment")
    if mkt_inst2:
        mkt_tot2 = res2["deposit"] + mkt_inst2 * int(nm2)
        over2    = (mkt_tot2 - res2["fair_total"]) / res2["fair_total"]
        tag2, tc2, _ = fairness_tag(over2)
        rd.markdown(f"""<div class="mcard">
          <div class="bar" style="background:{tc2}"></div>
          <div class="mlabel">Market Overpricing</div>
          <div class="mvalue" style="color:{tc2}">{over2*100:.1f}%</div>
          <div class="msub">{tag2}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # PDF for tab 2
    pd_low2  = max(0.0, pd2 * 0.9)
    pd_high2 = min(1.0, pd2 * 1.1)
    scen2 = [("Base Case",   pd2,     ad2, r2,       nm2),
             ("PD −10%",     pd_low2, ad2, r2,       nm2),
             ("PD +10%",     pd_high2,ad2, r2,       nm2),
             ("Admin 8%",    pd2,     8.0, r2,       nm2),
             ("Rate +2pp",   pd2,     ad2, r2+0.02,  nm2),
             ("Rate −1pp",   pd2,     ad2, max(0,r2-0.01), nm2),
             ("18 months",   pd2,     ad2, r2,       18),
             ("6 months",    pd2,     ad2, r2,       6)]
    rows2 = []
    for name, pd_s, adm, r_s, n_s in scen2:
        rr = fair_installment(cp2, dp2, adm, int(n_s), r_s, pd_s)
        rows2.append({
            "Scenario":           name,
            "PD":                 round(pd_s, 3),
            "Admin %":            float(adm),
            "r":                  round(r_s, 4),
            "Term (mo)":          int(n_s),
            "Fair Monthly (KSh)": round(rr["fair_monthly"], 2),
            "Fair Total (KSh)":   round(rr["fair_total"], 2),
        })
    sens2 = pd.DataFrame(rows2)

    mkt_info2 = {"provided": False}
    if mkt_inst2 and np.isfinite(over2):
        mkt_info2 = {
            "provided": True, "market_monthly": mkt_inst2,
            "market_total": mkt_tot2, "over_amt": mkt_tot2 - res2["fair_total"],
            "over_pct": over2,
            "fairness_score": max(0, min(100, 100 - over2*100)),
            "tag": tag2, "tag_explain": "Auto-detected from pasted text",
            "implied_apr": float("nan"),
        }

    dl2c, info2c = st.columns([0.3, 0.7])
    with dl2c:
        pdf2 = build_pdf(
            inputs={"cash_price": float(cp2), "n_months": int(nm2),
                    "deposit_pct": float(dp2), "admin_pct": float(ad2),
                    "r_monthly": float(r2), "income_ksh": float(inc2)},
            pd_value=float(pd2), radcf=res2,
            market=mkt_info2, sensitivity_df=sens2,
        )
        st.download_button(
            "⬇  Download RADCF Report (PDF)", data=pdf2,
            file_name=f"RADCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", key="dl_pdf2", use_container_width=True,
        )
    with info2c:
        st.markdown("""
        <div class="ibox">
        Text extraction uses regex pattern matching. Always review extracted values before
        computing. For best results, include keywords like
        <em>"cash price: KSh …"</em>, <em>"deposit X%"</em>, <em>"X months"</em>,
        <em>"installment KES …"</em>, <em>"admin fee X%"</em>.
        </div>""", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid #e2e8f0;
     text-align:center;color:#94a3b8;font-size:.77rem">
  <strong style="color:#64748b">RADCF Fair Pricing Engine</strong> ·
  Egerton University · Department of Mathematics · BSc Actuarial Science ·
  Model: KNBS KCHSP 2022 · β₀=1.5309, β₁=−0.7322 ·
  <em>For research and educational purposes only</em>
</div>""", unsafe_allow_html=True)
