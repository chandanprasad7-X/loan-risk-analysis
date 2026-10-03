"""
AI/ML Powered Loan Risk Assessment System
Interactive PowerBI Command Center & Multi-Module Application
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import calculate_dti, prepare_input_dataframe
from src.prediction import LoanRiskPredictor
from src.evaluate_model import evaluate_all_models

# Page Configuration
st.set_page_config(
    page_title="FinTech Risk Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark Theme CSS (Raw String to Prevent Syntax Errors)
CSS_STYLE = r"""
<style>
    .stApp {
        background-color: #0D0E15;
        color: #E2E8F0;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    .top-bar {
        background: #151824;
        padding: 12px 20px;
        border-radius: 10px;
        border: 1px solid #2A2E3F;
        margin-bottom: 15px;
    }
    .top-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .kpi-tile {
        background-color: #151824;
        border: 1px solid #2A2E3F;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 2px;
    }
    .glass-box {
        background-color: #151824;
        border: 1px solid #2A2E3F;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
    section[data-testid="stSidebar"] {
        background-color: #08090E !important;
        border-right: 1px solid #2A2E3F;
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    try:
        return LoanRiskPredictor(model_dir="model"), None
    except Exception as e:
        return None, str(e)


predictor, load_error = get_predictor()

# Load Dataset for Portfolio Analytics & Dashboard
DATA_PATH = "data/loan_data.csv"
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH)
else:
    np.random.seed(42)
    df_raw = pd.DataFrame({
        "Credit_Score": np.random.randint(400, 850, 200),
        "Risk_Label": np.random.choice([0, 1], 200, p=[0.7, 0.3]),
        "DTI": np.random.uniform(10, 65, 200),
        "Loan_Amount": np.random.randint(50000, 1000000, 200),
        "Monthly_Income": np.random.randint(25000, 200000, 200)
    })

# Sidebar Header & Navigation
st.sidebar.markdown("### ⚡ **FINTECH RISK AI**")
st.sidebar.caption("Interactive Executive Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "⚡ Interactive Command Center",
        "🔍 Real-time Underwriting",
        "🎛️ Dynamic 'What-If' Simulator",
        "📊 Portfolio Analytics",
        "🤖 XAI Model Intelligence",
        "📈 Performance Benchmark"
    ]
)

# Global Interactive Filters for Page 1
if page == "⚡ Interactive Command Center":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ **DASHBOARD FILTERS**")
    score_filter = st.sidebar.slider("Credit Score Range", 300, 850, (400, 850))
    dti_filter = st.sidebar.slider("Max DTI Threshold (%)", 0.0, 100.0, 65.0)
    income_filter = st.sidebar.number_input("Min Monthly Income (₹)", value=20000, step=5000)

    df_filtered = df_raw[
        (df_raw["Credit_Score"] >= score_filter[0]) &
        (df_raw["Credit_Score"] <= score_filter[1]) &
        (df_raw["DTI"] <= dti_filter) &
        (df_raw["Monthly_Income"] >= income_filter)
    ]
else:
    df_filtered = df_raw

# PAGE 1: INTERACTIVE POWERBI COMMAND CENTER
if page == "⚡ Interactive Command Center":
    total_apps = len(df_filtered)
    default_count = int(df_filtered["Risk_Label"].sum()) if total_apps > 0 and "Risk_Label" in df_filtered.columns else 0
    default_rate = (default_count / total_apps * 100) if total_apps > 0 else 0.0
    good_loans_pct = 100.0 - default_rate
    total_risk_val = (df_filtered[df_filtered["Risk_Label"] == 1]["Loan_Amount"].sum() / 1e6) if total_apps > 0 and "Loan_Amount" in df_filtered.columns else 0.0

    st.markdown("""
    <div class="top-bar">
        <div class="top-title">⚡ LOAN RISK ASSESSMENT COMMAND CENTER</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="kpi-tile"><div class="kpi-title">Good Loans %</div><div class="kpi-value" style="color:#22C55E;">{good_loans_pct:.2f}%</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-tile"><div class="kpi-title">Default Rate %</div><div class="kpi-value" style="color:#EF4444;">{default_rate:.2f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-tile"><div class="kpi-title">Portfolio Risk</div><div class="kpi-value" style="color:#A855F7;">₹{total_risk_val:.2f}M</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-tile"><div class="kpi-title">Filtered Applicants</div><div class="kpi-value" style="color:#38BDF8;">{total_apps:,}</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-tile"><div class="kpi-title">Deployed Model</div><div class="kpi-value" style="color:#F59E0B; font-size:1.1rem; margin-top:4px;">Random Forest</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if total_apps == 0:
        st.warning("⚠️ No records match the selected sidebar filters. Adjust sliders to view dynamic results.")
    else:
        c_left, c_mid, c_right = st.columns([1.2, 1.8, 1.2])

        with c_left:
            st.markdown("### 🎯 Portfolio Breakdown")
            risk_counts = df_filtered["Risk_Label"].value_counts().reset_index()
            risk_counts.columns = ["Status", "Count"]
            risk_counts["Status"] = risk_counts["Status"].map({0: "Approved (Low)", 1: "Default (High)"})

            fig_donut = px.pie(
                risk_counts, values="Count", names="Status",
                hole=0.55, color="Status",
                color_discrete_map={"Approved (Low)": "#22C55E", "Default (High)": "#A855F7"}
            )
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8', size=11),
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            st.markdown("##### Default Rate by DTI Band")
            df_filtered["DTI_Band"] = pd.cut(df_filtered["DTI"], bins=[0, 20, 40, 60, 100], labels=["<20%", "20-40%", "40-60%", ">60%"])
            dti_summary = df_filtered.groupby("DTI_Band", observed=False)["Risk_Label"].mean().reset_index()
            dti_summary["Risk_Label"] = (dti_summary["Risk_Label"] * 100).round(1).astype(str) + "%"
            dti_summary.columns = ["DTI Band", "Default Rate"]
            st.dataframe(dti_summary, hide_index=True, use_container_width=True)

        with c_mid:
            st.markdown("### 📈 Credit Band Analysis")
            df_filtered["Score_Band"] = pd.cut(
                df_filtered["Credit_Score"],
                bins=[300, 580, 670, 740, 850],
                labels=["Poor", "Fair", "Good", "Excellent"]
            )
            band_df = df_filtered.groupby("Score_Band", observed=False).agg(
                Total_Loans=("Risk_Label", "count"),
                Default_Rate=("Risk_Label", "mean")
            ).reset_index()
            band_df["Default_Rate_Pct"] = band_df["Default_Rate"] * 100

            fig_combo = go.Figure()
            fig_combo.add_trace(go.Bar(
                x=band_df["Score_Band"], y=band_df["Total_Loans"],
                name="Loan Volume", marker_color="#8B5CF6"
            ))
            fig_combo.add_trace(go.Scatter(
                x=band_df["Score_Band"], y=band_df["Default_Rate_Pct"],
                name="Default Rate %", yaxis="y2", mode="lines+markers",
                line=dict(color="#38BDF8", width=3)
            ))

            fig_combo.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'), margin=dict(t=20, b=20, l=10, r=10),
                yaxis=dict(title="Loan Volume", gridcolor="#2A2E3F"),
                yaxis2=dict(title="Default Rate %", overlaying="y", side="right", range=[0, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_combo, use_container_width=True)

            st.markdown("##### Income Distribution vs. Risk")
            fig_hist = px.histogram(
                df_filtered, x="Monthly_Income", color="Risk_Label",
                color_discrete_map={0: "#22C55E", 1: "#A855F7"}, nbins=20
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94A3B8'), margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c_right:
            st.markdown("### ⚡ Dynamic Evaluator")
            with st.form("quick_eval_form"):
                credit_input = st.slider("Credit Score", 300, 850, 720)
                income_input = st.number_input("Monthly Income (₹)", value=75000.0, step=5000.0)
                debt_input = st.number_input("Monthly Debt (₹)", value=18000.0, step=2000.0)
                amount_input = st.number_input("Loan Amount (₹)", value=400000.0, step=25000.0)
                defaults_input = st.selectbox("Previous Defaults", options=[0, 1, 2, 3, 4], index=0)

                calc_dti = calculate_dti(debt_input, income_input)
                st.caption(f"Calculated DTI: **{calc_dti:.2f}%**")

                eval_btn = st.form_submit_button("🔍 RUN UNDERWRITING")

            if eval_btn and predictor:
                eval_df = prepare_input_dataframe(
                    age=32, monthly_income=income_input, loan_amount=amount_input,
                    loan_term_months=36, credit_score=credit_input,
                    previous_defaults=defaults_input, monthly_debt_payment=debt_input, dti=calc_dti
                )
                res = predictor.predict(eval_df)

                st.markdown("---")
                st.markdown(f"#### Verdict: **<font color='{res['risk_color']}'>{res['raw_class_label']}</font>**", unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["risk_score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'color': '#FFFFFF', 'size': 14}},
                    number={'font': {'color': '#FFFFFF', 'size': 26}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                        'bar': {'color': res["risk_color"]},
                        'bgcolor': "#151824",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.2)"},
                            {'range': [30, 60], 'color': "rgba(234, 179, 8, 0.2)"},
                            {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=180, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

# PAGE 2: REAL-TIME UNDERWRITING & COPY OPTION
elif page == "🔍 Real-time Underwriting":
    st.markdown("## 🔍 Applicant Risk Assessment & Copy Report")
    st.caption("Enter financial parameters for immediate algorithmic credit evaluation")

    if load_error:
        st.error(f"⚠️ Model Artifact Error: {load_error}")
    else:
        with st.form("underwriting_form"):
            st.markdown("##### 👤 Applicant Profile")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Years)", min_value=18, max_value=90, value=34)
            with c2:
                defaults = st.number_input("Previous Credit Defaults", min_value=0, max_value=10, value=0)

            st.markdown("##### 💵 Income & Debt Profile")
            c3, c4, c5 = st.columns(3)
            with c3:
                income = st.number_input("Monthly Income (₹)", min_value=1000.0, value=85000.0, step=2500.0)
            with c4:
                debt = st.number_input("Monthly Debt Payments (₹)", min_value=0.0, value=18000.0, step=1000.0)
            with c5:
                credit = st.number_input("Credit Score (CIBIL)", min_value=300, max_value=850, value=750)

            st.markdown("##### 🏦 Requested Loan Parameters")
            c6, c7 = st.columns(2)
            with c6:
                amount = st.number_input("Loan Amount Requested (₹)", min_value=10000.0, value=450000.0, step=25000.0)
            with c7:
                term = st.selectbox("Loan Tenure (Months)", options=[12, 24, 36, 48, 60, 72, 120, 180, 240, 360], index=2)

            dti_calc = calculate_dti(debt, income)
            st.markdown(f"📈 **Calculated Debt-to-Income (DTI):** `<font color='#38BDF8'><b>{dti_calc:.2f}%</b></font>`", unsafe_allow_html=True)

            submit_btn = st.form_submit_button("⚡ RUN UNDERWRITING EVALUATION")

        if submit_btn:
            input_df = prepare_input_dataframe(
                age=age, monthly_income=income, loan_amount=amount,
                loan_term_months=term, credit_score=credit,
                previous_defaults=defaults, monthly_debt_payment=debt, dti=dti_calc
            )

            res = predictor.predict(input_df)

            st.markdown("---")
            st.markdown("### 📊 Decision Result")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("VERDICT", res["raw_class_label"])
            with m2:
                st.metric("RISK SCORE", f"{res['risk_score']} / 100")
            with m3:
                st.metric("DEFAULT PROBABILITY", f"{res['high_risk_probability']}%")
            with m4:
                st.metric("APPROVAL CAPACITY", f"{res['low_risk_probability']}%")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["risk_score"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Category: {res['presentation_category']}", 'font': {'color': '#F8FAFC', 'size': 18}},
                number={'font': {'color': '#F8FAFC', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94A3B8'},
                    'bar': {'color': res["risk_color"]},
                    'bgcolor': "#151824",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(234, 179, 8, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("### 📋 Copy Underwriting Report & Payload")
            summary_text = f"""=== FINTECH LOAN RISK UNDERWRITING REPORT ===
Verdict: {res['raw_class_label']} ({res['presentation_category']})
Risk Score: {res['risk_score']} / 100
Default Probability: {res['high_risk_probability']}%

Applicant Profile:
- Credit Score: {credit}
- DTI Ratio: {dti_calc:.2f}%
- Monthly Income: ₹{income:,.2f}
- Loan Amount: ₹{amount:,.2f}
- Previous Defaults: {defaults}
============================================="""

            json_payload = json.dumps({
                "verdict": res["raw_class_label"],
                "risk_score": res["risk_score"],
                "default_probability": res["high_risk_probability"],
                "inputs": {
                    "credit_score": credit,
                    "dti": round(dti_calc, 2),
                    "income": income,
                    "loan_amount": amount,
                    "defaults": defaults
                }
            }, indent=2)

            cp1, cp2 = st.columns(2)
            with cp1:
                st.caption("📋 Executive Text Summary (Hover block & click top-right icon to copy):")
                st.code(summary_text, language="text")
            with cp2:
                st.caption("📋 Audit JSON Payload (Hover block & click top-right icon to copy):")
                st.code(json_payload, language="json")

# PAGE 3: DYNAMIC WHAT-IF SIMULATOR
elif page == "🎛️ Dynamic 'What-If' Simulator":
    st.markdown("## 🎛️ Interactive What-If Simulator")
    st.caption("Adjust financial levers in real-time to simulate credit score sensitivity and risk migration")

    if load_error:
        st.error("Model unavailable.")
    else:
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_income = st.slider("Monthly Income (₹)", 15000, 250000, 75000, step=5000)
            sim_debt = st.slider("Monthly Debt Payments (₹)", 0, 120000, 20000, step=2000)
            sim_credit = st.slider("Credit Score (CIBIL)", 300, 850, 720, step=10)
        with sc2:
            sim_loan = st.slider("Requested Loan Amount (₹)", 50000, 3000000, 500000, step=25000)
            sim_defaults = st.slider("Previous Defaults", 0, 5, 0)
            sim_age = st.slider("Age", 20, 70, 32)

        sim_dti = calculate_dti(sim_debt, sim_income)
        sim_df = prepare_input_dataframe(
            age=sim_age, monthly_income=sim_income, loan_amount=sim_loan,
            loan_term_months=36, credit_score=sim_credit,
            previous_defaults=sim_defaults, monthly_debt_payment=sim_debt, dti=sim_dti
        )

        sim_res = predictor.predict(sim_df)

        st.markdown("---")
        st.markdown("### ⚡ Real-time Simulated Risk Score")

        sm1, sm2, sm3 = st.columns(3)
        with sm1:
            st.metric("SIMULATED RISK SCORE", f"{sim_res['risk_score']} / 100")
        with sm2:
            st.metric("RISK CATEGORY", sim_res["presentation_category"])
        with sm3:
            st.metric("COMPUTED DTI RATIO", f"{sim_dti:.2f}%")

        st.progress(sim_res["risk_score"] / 100.0)

# PAGE 4: PORTFOLIO ANALYTICS
elif page == "📊 Portfolio Analytics":
    st.markdown("## 📊 Loan Portfolio Analytics")
    st.caption("Exploratory distribution patterns across historical applicant records")

    if os.path.exists(DATA_PATH):
        st.dataframe(df_raw.head(10), use_container_width=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            fig_p = px.pie(
                df_raw, names="Risk_Label", title="Class Ratio (0=Approved/Low Risk, 1=High Risk)",
                color_discrete_sequence=["#22C55E", "#EF4444"], hole=0.4
            )
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
            st.plotly_chart(fig_p, use_container_width=True)
        with ac2:
            fig_h = px.histogram(
                df_raw, x="Credit_Score", color="Risk_Label",
                title="Credit Score Distribution vs. Risk Label", nbins=30,
                color_discrete_map={0: "#22C55E", 1: "#EF4444"}
            )
            fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
            st.plotly_chart(fig_h, use_container_width=True)

# PAGE 5: XAI MODEL INTELLIGENCE
elif page == "🤖 XAI Model Intelligence":
    st.markdown("## 🤖 Model Architecture & XAI")
    st.markdown("""
    <div class="glass-box">
        <h3>Model Technical Architecture</h3>
        <ul>
            <li><b>Ensemble Classifier:</b> Random Forest Classifier (150 trees, max depth = 10)</li>
            <li><b>Preprocessing:</b> Median SimpleImputer for handling missing entries without data leakage</li>
            <li><b>Feature Attributes:</b> Age, Monthly Income, Loan Amount, Loan Term, Credit Score, Previous Defaults, Monthly Debt Payment, DTI %</li>
            <li><b>Explainable AI Framework:</b> SHAP (SHapley Additive exPlanations) TreeExplainer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# PAGE 6: PERFORMANCE BENCHMARK
elif page == "📈 Performance Benchmark":
    st.markdown("## 📈 Comparative Benchmark")
    st.caption("Benchmark Random Forest against alternative machine learning classification models")

    if st.button("⚡ EXECUTE MULTI-MODEL BENCHMARK"):
        with st.spinner("Training and evaluating algorithms..."):
            res_df = evaluate_all_models()
            st.dataframe(res_df, use_container_width=True)

            fig_bm = px.bar(
                res_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
                x="Model", y="Score", color="Metric", barmode="group",
                title="Algorithm Performance Comparison (AUC-ROC, F1, Accuracy)",
                color_discrete_sequence=["#38BDF8", "#818CF8", "#22C55E", "#EAB308", "#EF4444"]
            )
            fig_bm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
            st.plotly_chart(fig_bm, use_container_width=True)