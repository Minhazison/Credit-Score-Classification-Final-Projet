import re
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "credit_model.pkl")

@st.cache_resource
def load_model():
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle
    except FileNotFoundError:
        return None

bundle = load_model()

# ── Helper functions (must match train_model.py exactly) ──────────────────────

def parse_credit_history_age(series):
    def _parse(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"(\d+)\s*Years?\s*(?:and)?\s*(\d+)\s*Months?", str(val))
        return (int(m.group(1)) * 12 + int(m.group(2))) if m else np.nan
    return series.apply(_parse)


def clean_single_row(row_dict, learned_stats):
    """Clean a single-row dict and return a cleaned DataFrame row."""
    df = pd.DataFrame([row_dict])

    # Numeric conversions & clamping
    for col in ["Age", "Annual_Income", "Num_of_Loan",
                "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("_", "", regex=False).str.strip(),
                errors="coerce"
            )

    # Clamp Age
    df.loc[(df["Age"] < 14) | (df["Age"] > 100), "Age"] = np.nan

    # Clamp bank accounts / cards
    df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
    df.loc[(df["Num_Credit_Card"]   < 0) | (df["Num_Credit_Card"]   > 25), "Num_Credit_Card"]   = np.nan
    df.loc[(df["Interest_Rate"]     < 1) | (df["Interest_Rate"]     > 50), "Interest_Rate"]     = np.nan
    df.loc[(df["Num_of_Loan"]       < 0) | (df["Num_of_Loan"]       > 25), "Num_of_Loan"]       = np.nan
    df["Delay_from_due_date"] = np.maximum(0, df["Delay_from_due_date"].fillna(0))
    df.loc[(df["Num_of_Delayed_Payment"] < 0) | (df["Num_of_Delayed_Payment"] > 50), "Num_of_Delayed_Payment"] = np.nan
    df.loc[(df["Num_Credit_Inquiries"] < 0) | (df["Num_Credit_Inquiries"] > 30), "Num_Credit_Inquiries"] = np.nan
    df.loc[df["Total_EMI_per_month"] > 2000, "Total_EMI_per_month"] = np.nan
    df.loc[df["Amount_invested_monthly"] >= 10000, "Amount_invested_monthly"] = np.nan
    df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan

    # Occupation / Credit_Mix / Payment_Behaviour sentinel cleanup
    df["Occupation"]        = df["Occupation"].replace(["_______", ""], np.nan)
    df["Credit_Mix"]        = df["Credit_Mix"].replace(["_", ""], np.nan)
    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace(["!@9#%8", ""], np.nan)

    # Credit History
    df["Credit_History_Months"] = parse_credit_history_age(
        df.get("Credit_History_Age", pd.Series(["0 Years and 0 Months"]))
    )

    # Payment of Min Amount
    df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map(
        {"No": 0, "NM": 0, "Yes": 1}
    ).fillna(0).astype(int)

    # Loan type flags
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Personal Loan",
                  "Home Equity Loan", "Mortgage Loan", "Student Loan",
                  "Debt Consolidation Loan", "Payday Loan"]
    loan_str = df.get("Type_of_Loan", pd.Series([""])).fillna("")
    for lt in loan_types:
        col_name = "Loan_" + lt.replace(" ", "_").replace("-", "_")
        df[col_name] = loan_str.str.contains(lt, regex=False).astype(int)

    # Fill remaining NaN with learned_stats
    for col, val in learned_stats.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

    # Credit Mix Score
    df["Credit_Mix_Score"] = df["Credit_Mix"].map(
        {"Bad": 0, "Standard": 1, "Good": 2}
    ).fillna(1).astype(int)

    return df


def engineer_features(df):
    data = df.copy()
    data["Debt_to_Income"]      = data["Outstanding_Debt"]        / (data["Annual_Income"]          + 1.0)
    data["EMI_to_Salary"]       = data["Total_EMI_per_month"]     / (data["Monthly_Inhand_Salary"]  + 1.0)
    data["Invest_to_Salary"]    = data["Amount_invested_monthly"] / (data["Monthly_Inhand_Salary"]  + 1.0)
    data["Savings_Ratio"]       = data["Monthly_Balance"]         / (data["Monthly_Inhand_Salary"]  + 1.0)
    data["Card_to_Account_Ratio"] = data["Num_Credit_Card"]       / (data["Num_Bank_Accounts"]      + 1.0)
    data["Delay_Ratio"]         = data["Num_of_Delayed_Payment"]  / ((data["Credit_History_Months"] / 12.0) + 1.0)
    return data


def predict(row_dict, bundle):
    learned_stats    = bundle["learned_stats"]
    categorical_cols = bundle["categorical_cols"]
    feature_cols     = bundle["feature_cols"]
    rf               = bundle["rf"]
    hgb              = bundle["hgb"]
    inv_target_mapping = bundle["inv_target_mapping"]

    df = clean_single_row(row_dict, learned_stats)
    df = engineer_features(df)

    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_enc.reindex(columns=feature_cols, fill_value=0).fillna(0)

    rf_p  = rf.predict_proba(X)
    hgb_p = hgb.predict_proba(X)
    ens_p = 0.60 * rf_p + 0.40 * hgb_p

    pred_idx  = int(ens_p.argmax(axis=1)[0])
    label     = inv_target_mapping[pred_idx]
    proba_dict = {inv_target_mapping[i]: float(ens_p[0, i]) for i in range(3)}
    return label, proba_dict


# ── Color helpers ─────────────────────────────────────────────────────────────

SCORE_COLORS = {"Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c"}

def score_badge(label):
    color = SCORE_COLORS.get(label, "#888")
    return f'<span style="background:{color};color:white;padding:6px 18px;border-radius:12px;font-size:1.3rem;font-weight:700">{label}</span>'


# ── App layout ────────────────────────────────────────────────────────────────

st.title("💳 Smart Credit Scoring Dashboard")
st.markdown("*Powered by a soft-voting ensemble of **Random Forest** + **HistGradientBoosting***")

if bundle is None:
    st.error(
        f"❌  Model file **`{MODEL_PATH}`** not found.\n\n"
        "Run  `python train_model.py`  first (requires `train.csv` in the same folder)."
    )
    st.stop()

st.success(" Model loaded successfully")

# ── Sidebar: input form ───────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Customer Information")

    occupation_options = [
        "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer",
        "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager",
        "Accountant", "Musician", "Mechanic", "Writer", "Architect", "Other"
    ]
    credit_mix_options     = ["Good", "Standard", "Bad"]
    payment_beh_options    = [
        "High_spent_Large_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Small_value_payments",
        "Low_spent_Large_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Small_value_payments",
    ]
    pay_min_options        = ["Yes", "No", "NM"]
    credit_hist_years      = st.slider("Credit History (Years)", 0, 40, 5)
    credit_hist_months_extra = st.slider("…and Months", 0, 11, 0)
    credit_hist_age_str    = f"{credit_hist_years} Years and {credit_hist_months_extra} Months"

    loan_type_options = [
        "Auto Loan", "Credit-Builder Loan", "Personal Loan",
        "Home Equity Loan", "Mortgage Loan", "Student Loan",
        "Debt Consolidation Loan", "Payday Loan"
    ]
    selected_loans = st.multiselect("Type of Loans", loan_type_options, default=[])
    type_of_loan_str = ", ".join(selected_loans) if selected_loans else ""

    age               = st.number_input("Age",                      14, 100, 32)
    annual_income     = st.number_input("Annual Income ($)",         0.0, 1e7, 48000.0, step=1000.0)
    monthly_salary    = st.number_input("Monthly Inhand Salary ($)", 0.0, 1e6, annual_income / 12, step=100.0)
    num_bank_accounts = st.slider("Number of Bank Accounts",         0, 20, 4)
    num_credit_cards  = st.slider("Number of Credit Cards",          0, 25, 4)
    interest_rate     = st.slider("Interest Rate (%)",               1, 50, 13)
    num_loans         = st.slider("Number of Loans",                 0, 25, 2)
    delay_due_date    = st.slider("Days Delayed from Due Date",      0, 100, 5)
    num_delayed_pay   = st.slider("Number of Delayed Payments",      0, 50, 3)
    changed_credit_lim= st.number_input("Changed Credit Limit",     -20.0, 100.0, 7.0)
    num_credit_inq    = st.slider("Number of Credit Inquiries",      0, 30, 4)
    credit_mix        = st.selectbox("Credit Mix",                   credit_mix_options)
    outstanding_debt  = st.number_input("Outstanding Debt ($)",      0.0, 50000.0, 1200.0, step=50.0)
    credit_util_ratio = st.slider("Credit Utilisation Ratio (%)",    0.0, 100.0, 30.0)
    pay_min           = st.selectbox("Payment of Min Amount",        pay_min_options)
    total_emi         = st.number_input("Total EMI per Month ($)",   0.0, 2000.0, 80.0, step=5.0)
    amount_invested   = st.number_input("Amount Invested Monthly ($)",0.0, 9999.0, 150.0, step=10.0)
    payment_beh       = st.selectbox("Payment Behaviour",            payment_beh_options)
    monthly_balance   = st.number_input("Monthly Balance ($)",       0.0, 50000.0, 400.0, step=10.0)
    occupation        = st.selectbox("Occupation",                   occupation_options)

    predict_btn = st.button("🔍 Predict Credit Score", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Summary")
    summary = {
        "Age": age,
        "Annual Income": f"${annual_income:,.0f}",
        "Monthly Salary": f"${monthly_salary:,.0f}",
        "Bank Accounts": num_bank_accounts,
        "Credit Cards": num_credit_cards,
        "Interest Rate": f"{interest_rate}%",
        "Loans": num_loans,
        "Delayed Payments": num_delayed_pay,
        "Credit Mix": credit_mix,
        "Outstanding Debt": f"${outstanding_debt:,.0f}",
        "Credit History": credit_hist_age_str,
        "Occupation": occupation,
    }
    for k, v in summary.items():
        st.markdown(f"**{k}:** {v}")

with col2:
    if predict_btn:
        row = {
            "Customer_ID":              "LIVE_PRED",
            "Age":                      age,
            "Occupation":               occupation,
            "Annual_Income":            annual_income,
            "Monthly_Inhand_Salary":    monthly_salary,
            "Num_Bank_Accounts":        num_bank_accounts,
            "Num_Credit_Card":          num_credit_cards,
            "Interest_Rate":            interest_rate,
            "Num_of_Loan":              num_loans,
            "Type_of_Loan":             type_of_loan_str,
            "Delay_from_due_date":      delay_due_date,
            "Num_of_Delayed_Payment":   num_delayed_pay,
            "Changed_Credit_Limit":     changed_credit_lim,
            "Num_Credit_Inquiries":     num_credit_inq,
            "Credit_Mix":               credit_mix,
            "Outstanding_Debt":         outstanding_debt,
            "Credit_Utilization_Ratio": credit_util_ratio,
            "Credit_History_Age":       credit_hist_age_str,
            "Payment_of_Min_Amount":    pay_min,
            "Total_EMI_per_month":      total_emi,
            "Amount_invested_monthly":  amount_invested,
            "Payment_Behaviour":        payment_beh,
            "Monthly_Balance":          monthly_balance,
        }

        with st.spinner("Running ensemble prediction…"):
            label, proba_dict = predict(row, bundle)

        st.subheader(" Prediction Result")
        st.markdown(score_badge(label), unsafe_allow_html=True)
        st.markdown("")

        # Probability bar chart
        labels = list(proba_dict.keys())
        probs  = [proba_dict[l] * 100 for l in labels]
        colors = [SCORE_COLORS[l] for l in labels]

        fig, ax = plt.subplots(figsize=(5, 2.8))
        bars = ax.barh(labels, probs, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Ensemble Probability Distribution")
        ax.set_xlim(0, 100)
        for bar, p in zip(bars, probs):
            ax.text(p + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Class Probabilities**")
        for lbl, p in proba_dict.items():
            st.progress(p, text=f"{lbl}: {p*100:.1f}%")
    else:
        st.info("  Fill in the form on the left and click **Predict Credit Score**")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Smart Credit Scoring | Ensemble: 60% Random Forest + 40% HistGradientBoosting")
