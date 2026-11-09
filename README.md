#  Smart Credit: Automating Financial Risk Brackets

An end-to-end Machine Learning pipeline and interactive Streamlit web dashboard for credit risk classification. The system predicts credit score brackets (**Good**, **Standard**, or **Poor**) using a soft-voting ensemble of **Random Forest** and **HistGradientBoosting**.
[The credit_model.pkl file you will get after running the Smart_Credit_Scoring_Model.ipynb file. The data file is Bank_data.zip.]
---

##  Project Overview

Traditional credit scoring systems often struggle with noisy data, non-linear relationships, and complex financial behavior patterns. This project cleans and standardizes messy  credit records, performs automated domain-specific feature engineering, trains ensemble classifiers, and serves real-time inference via a user-friendly Streamlit interface.

### Key Highlights
- **Target Classes**: `Poor`, `Standard`, `Good`
- **Model Architecture**: Soft-voting ensemble combining:
  - **Random Forest Classifier** ($60\%$ weight)
  - **HistGradientBoosting Classifier** ($40\%$ weight)
- **Validation Accuracy**: $\approx 79.14\%$
- **Interactive UI**: Real-time scoring, input validation, and probability distribution charts built with Streamlit and Matplotlib.

---

##  Project Structure

```text
.
├── main.py              # Streamlit dashboard application for live predictions
├── train_model.py       # Standalone training script to produce credit_model.pkl
├── credit_model.pkl     # Serialized ensemble model and preprocessor bundle (generated)
├── requirements.txt     # Python dependencies
├── train.csv            # Training dataset (Bank data)
├── test.csv             # Evaluation/test dataset
└── README.md            # Project documentation
```

---

##  Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10+ recommended)
- `pip` or virtual environment manager (`venv` / `conda`)

### 2. Clone Repository & Create Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Training the Model

Before running the application for the first time, you must train the model and generate the serialized bundle (`credit_model.pkl`).

Make sure `train.csv` is present in the project directory, then run:

```bash
python train_model.py
```

### What happens during training:
1. **Data Cleaning**: Strips special characters, parses strings, handles negative values, clamps outliers, and performs customer-level forward/backward filling.
2. **Feature Engineering**: Derives key financial ratios:
   - `Debt_to_Income = Outstanding_Debt / (Annual_Income + 1.0)`
   - `EMI_to_Salary = Total_EMI_per_month / (Monthly_Inhand_Salary + 1.0)`
   - `Invest_to_Salary = Amount_invested_monthly / (Monthly_Inhand_Salary + 1.0)`
   - `Savings_Ratio = Monthly_Balance / (Monthly_Inhand_Salary + 1.0)`
   - `Card_to_Account_Ratio = Num_Credit_Card / (Num_Bank_Accounts + 1.0)`
   - `Delay_Ratio = Num_of_Delayed_Payment / ((Credit_History_Months / 12.0) + 1.0)`
3. **Encoding & Alignment**: One-hot encodes categorical features (`Occupation`, `Credit_Mix`, `Payment_Behaviour`).
4. **Ensemble Training**: Trains `RandomForestClassifier` and `HistGradientBoostingClassifier`.
5. **Artifact Bundling**: Saves models, learned statistical medians, categorical mappings, and column signatures to `credit_model.pkl`.

---

##  Running the Streamlit Dashboard

Once `credit_model.pkl` is generated, launch the interactive web dashboard:

```bash
streamlit run main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Dashboard Features:
- **Customer Information Sidebar**: Enter demographic, employment, account, and debt parameters.
- **Dynamic Summaries**: Review key financial metrics before evaluation.
- **Instant Credit Risk Prediction**: Displays the predicted credit bracket (`Good`, `Standard`, `Poor`) with high-visibility status indicators.
- **Probability Breakdown**: Provides real-time probability charts showing the ensemble's confidence score for each bracket.

---

##  Model Architecture & Performance

| Model | Hyperparameters / Setup | Role |
| :--- | :--- | :--- |
| **Random Forest** | `n_estimators=200`, `max_depth=20`, `min_samples_leaf=5` | Captures high-dimensional feature interactions and limits overfitting. |
| **HistGradientBoosting** | `max_iter=300`, `max_depth=8`, `learning_rate=0.05` | Fast, binned gradient boosting suited for numerical variances. |
| **Ensemble** | $P_{\text{ens}} = 0.60 \times P_{\text{RF}} + 0.40 \times P_{\text{HGB}}$ | Soft-voting prediction outputting final class probabilities. |

---

##  Requirements

- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `streamlit >= 1.28.0`
- `joblib >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
