# =============================================================================
# Smart Credit Scoring Model – Model Export Script
# Run this script ONCE to train & save the model as credit_model.pkl
# Requires train.csv in the same directory (from Bank_data.zip)
# =============================================================================

import os, re, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_credit_history_age(series):
    def _parse(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"(\d+)\s*Years?\s*(?:and)?\s*(\d+)\s*Months?", str(val))
        return (int(m.group(1)) * 12 + int(m.group(2))) if m else np.nan
    return series.apply(_parse)


def clean_credit_data(df, is_train=True, learned_stats=None):
    data = df.copy()

    # 1. Age
    s = data["Age"].astype(str).str.replace("_", "", regex=False).str.strip()
    age_num = pd.to_numeric(s, errors="coerce")
    age_num.loc[(age_num < 14) | (age_num > 100)] = np.nan
    data["Age"] = age_num
    data["Age"] = data.groupby("Customer_ID", sort=False)["Age"].ffill().bfill()

    # 2. Occupation
    data["Occupation"] = data["Occupation"].replace(["_______", ""], np.nan)
    data["Occupation"] = data.groupby("Customer_ID", sort=False)["Occupation"].ffill().bfill()

    # 3. Annual Income
    s = data["Annual_Income"].astype(str).str.replace("_", "", regex=False).str.strip()
    data["Annual_Income"] = pd.to_numeric(s, errors="coerce")
    data["Annual_Income"] = data.groupby("Customer_ID", sort=False)["Annual_Income"].ffill().bfill()

    # 4. Monthly Inhand Salary
    data["Monthly_Inhand_Salary"] = data.groupby("Customer_ID", sort=False)["Monthly_Inhand_Salary"].ffill().bfill()
    data["Monthly_Inhand_Salary"] = data["Monthly_Inhand_Salary"].fillna(data["Annual_Income"] / 12.0)

    # 5. Num Bank Accounts
    data.loc[(data["Num_Bank_Accounts"] < 0) | (data["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
    data["Num_Bank_Accounts"] = data.groupby("Customer_ID", sort=False)["Num_Bank_Accounts"].ffill().bfill()

    # 6. Num Credit Cards
    data.loc[(data["Num_Credit_Card"] < 0) | (data["Num_Credit_Card"] > 25), "Num_Credit_Card"] = np.nan
    data["Num_Credit_Card"] = data.groupby("Customer_ID", sort=False)["Num_Credit_Card"].ffill().bfill()

    # 7. Interest Rate
    data.loc[(data["Interest_Rate"] < 1) | (data["Interest_Rate"] > 50), "Interest_Rate"] = np.nan
    data["Interest_Rate"] = data.groupby("Customer_ID", sort=False)["Interest_Rate"].ffill().bfill()

    # 8. Num of Loan
    s = data["Num_of_Loan"].astype(str).str.replace("_", "", regex=False).str.strip()
    n_loan = pd.to_numeric(s, errors="coerce")
    n_loan.loc[(n_loan < 0) | (n_loan > 25)] = np.nan
    data["Num_of_Loan"] = n_loan
    data["Num_of_Loan"] = data.groupby("Customer_ID", sort=False)["Num_of_Loan"].ffill().bfill()

    # 9. Delay from due date
    data["Delay_from_due_date"] = np.maximum(0, data["Delay_from_due_date"])

    # 10. Num of Delayed Payment
    s = data["Num_of_Delayed_Payment"].astype(str).str.replace("_", "", regex=False).str.strip()
    n_del = pd.to_numeric(s, errors="coerce")
    n_del.loc[(n_del < 0) | (n_del > 50)] = np.nan
    data["Num_of_Delayed_Payment"] = n_del
    data["Num_of_Delayed_Payment"] = data.groupby("Customer_ID", sort=False)["Num_of_Delayed_Payment"].ffill().bfill()

    # 11. Changed Credit Limit
    s = data["Changed_Credit_Limit"].astype(str).str.replace("_", "", regex=False).str.strip()
    data["Changed_Credit_Limit"] = pd.to_numeric(s, errors="coerce")
    data["Changed_Credit_Limit"] = data.groupby("Customer_ID", sort=False)["Changed_Credit_Limit"].ffill().bfill()

    # 12. Num Credit Inquiries
    data.loc[(data["Num_Credit_Inquiries"] < 0) | (data["Num_Credit_Inquiries"] > 30), "Num_Credit_Inquiries"] = np.nan
    data["Num_Credit_Inquiries"] = data.groupby("Customer_ID", sort=False)["Num_Credit_Inquiries"].ffill().bfill()

    # 13. Credit Mix
    data["Credit_Mix"] = data["Credit_Mix"].replace(["_", ""], np.nan)
    data["Credit_Mix"] = data.groupby("Customer_ID", sort=False)["Credit_Mix"].ffill().bfill()

    # 14. Outstanding Debt
    s = data["Outstanding_Debt"].astype(str).str.replace("_", "", regex=False).str.strip()
    data["Outstanding_Debt"] = pd.to_numeric(s, errors="coerce")
    data["Outstanding_Debt"] = data.groupby("Customer_ID", sort=False)["Outstanding_Debt"].ffill().bfill()

    # 15. Credit History Age
    data["Credit_History_Months"] = parse_credit_history_age(data["Credit_History_Age"])
    data["Credit_History_Months"] = data.groupby("Customer_ID", sort=False)["Credit_History_Months"].ffill().bfill()

    # 16. Payment of Min Amount
    data["Payment_of_Min_Amount"] = data["Payment_of_Min_Amount"].map({"No": 0, "NM": 0, "Yes": 1}).fillna(0).astype(int)

    # 17. Total EMI per month
    data.loc[data["Total_EMI_per_month"] > 2000, "Total_EMI_per_month"] = np.nan
    data["Total_EMI_per_month"] = data.groupby("Customer_ID", sort=False)["Total_EMI_per_month"].ffill().bfill()

    # 18. Amount invested monthly
    s = data["Amount_invested_monthly"].astype(str).str.replace("_", "", regex=False).str.strip()
    inv_num = pd.to_numeric(s, errors="coerce")
    inv_num.loc[inv_num >= 10000] = np.nan
    data["Amount_invested_monthly"] = inv_num
    data["Amount_invested_monthly"] = data.groupby("Customer_ID", sort=False)["Amount_invested_monthly"].ffill().bfill()

    # 19. Payment Behaviour
    data["Payment_Behaviour"] = data["Payment_Behaviour"].replace(["!@9#%8", ""], np.nan)
    data["Payment_Behaviour"] = data.groupby("Customer_ID", sort=False)["Payment_Behaviour"].ffill().bfill()

    # 20. Monthly Balance
    s = data["Monthly_Balance"].astype(str).str.replace("_", "", regex=False).str.strip()
    bal_num = pd.to_numeric(s, errors="coerce")
    bal_num.loc[bal_num < 0] = np.nan
    data["Monthly_Balance"] = bal_num
    data["Monthly_Balance"] = data.groupby("Customer_ID", sort=False)["Monthly_Balance"].ffill().bfill()

    # Loan type flags
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
                  "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan"]
    loan_str = data["Type_of_Loan"].fillna("")
    for lt in loan_types:
        col_name = "Loan_" + lt.replace(" ", "_").replace("-", "_")
        data[col_name] = loan_str.str.contains(lt, regex=False).astype(int)

    # Fallback fill stats
    if is_train:
        learned_stats = {
            "Age": data["Age"].median(),
            "Occupation": "Other",
            "Annual_Income": data["Annual_Income"].median(),
            "Monthly_Inhand_Salary": data["Monthly_Inhand_Salary"].median(),
            "Num_Bank_Accounts": data["Num_Bank_Accounts"].median(),
            "Num_Credit_Card": data["Num_Credit_Card"].median(),
            "Interest_Rate": data["Interest_Rate"].median(),
            "Num_of_Loan": data["Num_of_Loan"].median(),
            "Num_of_Delayed_Payment": data["Num_of_Delayed_Payment"].median(),
            "Changed_Credit_Limit": data["Changed_Credit_Limit"].median(),
            "Num_Credit_Inquiries": data["Num_Credit_Inquiries"].median(),
            "Credit_Mix": "Standard",
            "Outstanding_Debt": data["Outstanding_Debt"].median(),
            "Credit_History_Months": data["Credit_History_Months"].median(),
            "Total_EMI_per_month": data["Total_EMI_per_month"].median(),
            "Amount_invested_monthly": data["Amount_invested_monthly"].median(),
            "Payment_Behaviour": "Low_spent_Small_value_payments",
            "Monthly_Balance": data["Monthly_Balance"].median(),
        }

    for col, val in learned_stats.items():
        if col in data.columns:
            data[col] = data[col].fillna(val)

    data["Credit_Mix_Score"] = data["Credit_Mix"].map({"Bad": 0, "Standard": 1, "Good": 2}).fillna(1).astype(int)

    return data, learned_stats


def engineer_financial_features(df):
    data = df.copy()
    data["Debt_to_Income"] = data["Outstanding_Debt"] / (data["Annual_Income"] + 1.0)
    data["EMI_to_Salary"] = data["Total_EMI_per_month"] / (data["Monthly_Inhand_Salary"] + 1.0)
    data["Invest_to_Salary"] = data["Amount_invested_monthly"] / (data["Monthly_Inhand_Salary"] + 1.0)
    data["Savings_Ratio"] = data["Monthly_Balance"] / (data["Monthly_Inhand_Salary"] + 1.0)
    data["Card_to_Account_Ratio"] = data["Num_Credit_Card"] / (data["Num_Bank_Accounts"] + 1.0)
    data["Delay_Ratio"] = data["Num_of_Delayed_Payment"] / ((data["Credit_History_Months"] / 12.0) + 1.0)
    return data


# ── Main training ─────────────────────────────────────────────────────────────

def train_and_save(train_csv="train.csv", output_pkl="credit_model.pkl"):
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(train_csv):
        train_csv = os.path.join(script_dir, train_csv)
    if not os.path.isabs(output_pkl):
        output_pkl = os.path.join(script_dir, output_pkl)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"\n❌  train.csv not found at:\n   {train_csv}\n\n"
            "👉  Copy train.csv (from Bank_data.zip) into the same folder as train_model.py"
        )

    print(f"Loading {train_csv} ...")
    train_df = pd.read_csv(train_csv, low_memory=False)
    
    # Target mapping
    target_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
    inv_target_mapping = {v: k for k, v in target_mapping.items()}

    # Clean + engineer
    train_clean, learned_stats = clean_credit_data(train_df, is_train=True)
    train_clean = engineer_financial_features(train_clean)

    # Encode categoricals
    categorical_cols = ["Occupation", "Credit_Mix", "Payment_Behaviour"]
    train_encoded = pd.get_dummies(train_clean, columns=categorical_cols, drop_first=True)

    # Drop non-feature columns
    drop_cols = ["ID", "Customer_ID", "Name", "SSN", "Month",
                 "Type_of_Loan", "Credit_History_Age", "Credit_Score"]
    feature_cols = [c for c in train_encoded.columns if c not in drop_cols]

    X = train_encoded[feature_cols].fillna(0)
    y = train_clean["Credit_Score"].map(target_mapping)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                       stratify=y, random_state=42)

    print("Training Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("Training HistGradientBoosting …")
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=8,
                                         learning_rate=0.05, random_state=42)
    hgb.fit(X_train, y_train)

    # Bundle everything needed for inference
    bundle = {
        "rf": rf,
        "hgb": hgb,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "learned_stats": learned_stats,
        "target_mapping": target_mapping,
        "inv_target_mapping": inv_target_mapping,
    }

    joblib.dump(bundle, output_pkl)
    print(f"\n✅  Model bundle saved → {output_pkl}")

    # Quick val accuracy
    rf_p  = rf.predict_proba(X_val)
    hgb_p = hgb.predict_proba(X_val)
    ens_p = 0.60 * rf_p + 0.40 * hgb_p
    preds = ens_p.argmax(axis=1)
    acc   = (preds == y_val.values).mean()
    print(f"   Ensemble val accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    train_and_save()
