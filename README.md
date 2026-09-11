# 📊 Vendor Risk Scorecard

<p align="center">

### Machine Learning-Based Supplier Risk Assessment & Procurement Decision Support

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![GitHub](https://img.shields.io/badge/Repository-GitHub-black?logo=github)

</p>

---

## 📌 Project Overview

**Vendor Risk Scorecard** is a Machine Learning-based supplier evaluation system developed to assess vendor performance and identify suppliers with higher operational risk.

The system analyzes important supplier performance indicators such as:

- 📦 Defect Rate
- 🚚 Delivery Performance
- ✅ Compliance
- 💰 Price Savings

These indicators are used to calculate risk-related features and classify vendor/order records into **Low Risk** and **High Risk** categories.

The project also uses **SHAP (SHapley Additive exPlanations)** to explain the Machine Learning predictions.

An interactive **Streamlit application** is developed to allow users to enter vendor performance information and receive an instant risk assessment.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Analyze supplier performance data.
- Identify supplier quality issues.
- Measure supplier defect rates.
- Analyze delivery performance.
- Evaluate supplier compliance.
- Calculate pricing and savings indicators.
- Engineer vendor risk indicators.
- Develop a Machine Learning classification model.
- Evaluate model performance.
- Rank vendors according to risk.
- Generate procurement recommendations.
- Apply Explainable AI using SHAP.
- Develop an interactive Streamlit dashboard.
- Deploy the application using Streamlit Cloud.

---

# 🔄 Project Workflow

```text
                    ┌─────────────────────┐
                    │  Supplier Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Cleaning    │
                    │ Missing Values      │
                    │ Invalid Records     │
                    │ Data Validation     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Defect Rate         │
                    │ Delivery Days       │
                    │ Price Savings       │
                    │ Compliance          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Risk Indicators    │
                    │ High Defect         │
                    │ Late Delivery       │
                    │ Low Compliance      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Logistic Regression │
                    │ Classification      │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Model Evaluation │      │   SHAP Analysis  │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Vendor Risk Ranking │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Procurement Actions │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘
```

---

# 📊 Dataset

The project uses a supplier purchase-order dataset containing:

- **777 records**
- **11 original attributes**
- **5 suppliers**
- **5 item categories**

### Original Dataset Features

| Feature | Description |
|---|---|
| `PO_ID` | Purchase order identifier |
| `Supplier` | Supplier/vendor name |
| `Order_Date` | Purchase order date |
| `Delivery_Date` | Delivery date |
| `Item_Category` | Category of purchased item |
| `Order_Status` | Current order status |
| `Quantity` | Ordered quantity |
| `Unit_Price` | Original unit price |
| `Negotiated_Price` | Negotiated purchase price |
| `Defective_Units` | Number of defective units |
| `Compliance` | Supplier compliance status |

### 📌 Dataset Source

The dataset used in this project was prepared/provided as the working supplier performance dataset for the academic project.

The dataset was processed and analyzed using Python and Pandas.

---

# 🧹 Data Cleaning

The dataset was checked and cleaned before performing Machine Learning.

### Data Cleaning Activities

- ✅ Converted `Order_Date` and `Delivery_Date` into datetime format.
- ✅ Identified missing delivery dates.
- ✅ Identified missing defective-unit values.
- ✅ Created missing-data indicators.
- ✅ Filled missing defective-unit values with zero while retaining a missing-data indicator.
- ✅ Identified invalid delivery records.
- ✅ Corrected records where delivery occurred before the order date.
- ✅ Checked for negative quantities.
- ✅ Checked for negative prices.
- ✅ Checked whether defective units were greater than ordered quantity.
- ✅ Checked for duplicate records.

### Final Dataset

```text
Rows                  : 777
Columns               : 17
Duplicate Records     : 0
Missing Defective Data: 0
Negative Delivery Days: 0
```

---

# ⚙️ Feature Engineering

Several performance indicators were created from the original dataset.

## 1. 🚚 Delivery Days

Delivery duration was calculated as:

```text
Delivery Days = Delivery Date - Order Date
```

This measures the time taken by a supplier to deliver an order.

---

## 2. 🏭 Defect Rate

Defect rate was calculated as:

```text
Defect Rate = Defective Units / Quantity
```

A higher defect rate indicates poorer product quality.

---

## 3. 💰 Price Difference

Price difference was calculated as:

```text
Price Difference =
Unit Price - Negotiated Price
```

---

## 4. 💵 Price Savings Percentage

Price savings was calculated as:

```text
Price Savings % =
((Unit Price - Negotiated Price) / Unit Price) × 100
```

This represents the percentage of savings achieved through negotiation.

---

# 🚨 Risk Assessment

Three risk indicators were created.

| Risk Indicator | Condition |
|---|---|
| 🔴 High Defect Risk | Defect Rate > 5% |
| 🔴 Compliance Risk | Compliance = No |
| 🔴 Late Delivery Risk | Delivery Days > 15 |

These thresholds were defined as project assumptions.

---

# 📈 Risk Score

The overall risk score was calculated using three risk indicators:

```text
Risk Score =
High Defect Flag
+
Low Compliance Flag
+
Late Delivery Flag
```

### Risk Score Interpretation

| Score | Interpretation |
|---:|---|
| 0 | No identified risk indicators |
| 1 | One risk indicator |
| 2 | Two risk indicators |
| 3 | Three risk indicators |

A record is classified as **High Risk** when:

```text
Risk Score >= 2
```

---

# 🤖 Machine Learning Model

## Algorithm

The project uses:

**Logistic Regression**

The model performs binary classification:

```text
0 → Low Risk
1 → High Risk
```

### Model Features

The Machine Learning model uses four main features:

```text
Defect_Rate
Delivery_Days
Price_Savings_Percent
Compliance
```

The risk flags were not directly used as model inputs because they were involved in creating the target variable. This helps prevent target leakage.

---

# 📚 Train-Test Split

The dataset was divided into training and testing datasets.

| Dataset | Records |
|---|---:|
| Training Dataset | 621 |
| Testing Dataset | 156 |

### Configuration

```text
Training Size : 80%
Testing Size  : 20%
Random State  : 42
```

`StandardScaler` was used to standardize the model features.

---

# 🏆 Model Performance

The Logistic Regression model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| 🎯 Accuracy | **93.59%** |
| Precision | **89.66%** |
| Recall | **78.79%** |
| F1 Score | **83.87%** |

### Confusion Matrix

```text
                    Predicted
                  Low Risk  High Risk
Actual Low Risk      120        3
Actual High Risk      7        26
```

The model correctly classified:

**146 out of 156 test records.**

---

# 🏅 Vendor Risk Ranking

Vendor-level performance was analyzed using average risk score and high-risk order percentage.

| Rank | Supplier | Risk Category |
|---:|---|---|
| 🥇 1 | **Delta_Logistics** | 🔴 High Risk |
| 🥈 2 | **Beta_Supplies** | 🔴 High Risk |
| 🥉 3 | **Gamma_Co** | 🟠 Medium Risk |
| 4 | **Epsilon_Group** | 🟢 Low Risk |
| 5 | **Alpha_Inc** | 🟢 Low Risk |

---

# 🔎 Vendor Risk Analysis

### 🔴 Delta_Logistics

Delta_Logistics has the highest overall risk.

Main concerns:

- High average defect rate
- Lowest compliance rate
- Highest average risk score
- Highest percentage of high-risk orders

**Procurement Action:**  
High-priority monitoring and improvement of quality and compliance should be considered before increasing orders.

---

### 🔴 Beta_Supplies

Beta_Supplies also shows relatively high operational risk.

Main concerns:

- High defect rate
- Lower compliance performance
- High percentage of high-risk orders

**Procurement Action:**  
Closely monitor quality and compliance performance.

---

### 🟠 Gamma_Co

Gamma_Co has a moderate risk profile.

**Procurement Action:**  
Maintain compliance and focus on improving defect control.

---

### 🟢 Epsilon_Group

Epsilon_Group demonstrates strong supplier performance.

**Procurement Action:**  
Preferred supplier; maintain current quality and compliance performance.

---

### 🟢 Alpha_Inc

Alpha_Inc shows strong quality performance and low overall risk.

**Procurement Action:**  
Preferred supplier for stable procurement performance.

---

# 💼 Procurement Recommendations

| Supplier | Recommendation |
|---|---|
| 🔴 Delta_Logistics | High-priority monitoring; improve quality control and compliance before increasing orders. |
| 🔴 Beta_Supplies | Closely monitor performance and focus on reducing defects and improving compliance. |
| 🟠 Gamma_Co | Moderate monitoring; improve defect control while maintaining compliance. |
| 🟢 Epsilon_Group | Preferred supplier; maintain current performance and compliance. |
| 🟢 Alpha_Inc | Preferred supplier due to strong quality performance and low overall risk. |

---

# 🔍 Explainable AI — SHAP

**SHAP (SHapley Additive exPlanations)** was used to make the Machine Learning model more interpretable.

### SHAP Feature Importance

The analysis identified the following order of feature influence:

```text
1. Delivery Days
2. Compliance
3. Defect Rate
4. Price Savings Percentage
```

### Key Finding

> **Delivery performance, compliance, and product quality have a stronger influence on model risk predictions than price savings.**

SHAP was also used to explain an individual prediction and identify how each feature contributed to the final model output.

---

# 🖥️ Streamlit Application

An interactive Streamlit web application was developed for vendor risk assessment.

### User Inputs

The application accepts:

```text
📦 Defect Rate (%)
🚚 Delivery Days
💰 Price Savings (%)
✅ Compliance
```

### Application Output

The application provides:

- 🟢 LOW RISK prediction
- 🔴 HIGH RISK prediction
- 📊 Model-estimated high-risk probability
- 🏆 Vendor risk ranking
- 📋 Procurement recommendations

---

# 📸 Application Preview

Screenshots of the Streamlit application can be added here after deployment.

```text
Add your Streamlit dashboard screenshots here.
```

Example:

```markdown
![Vendor Risk Dashboard](screenshots/dashboard.png)
```

---

# 🧪 Example Vendor Profiles

## 🟢 Example 1 — Low Risk

```text
Defect Rate     : 2%
Delivery Days   : 8
Price Savings   : 8%
Compliance      : Yes
```

Expected result:

**LOW RISK**

---

## 🔴 Example 2 — High Risk

```text
Defect Rate     : 12%
Delivery Days   : 20
Price Savings   : 5%
Compliance      : No
```

Expected result:

**HIGH RISK**

---

## 🟠 Example 3 — Moderate Performance

```text
Defect Rate     : 6%
Delivery Days   : 12
Price Savings   : 8%
Compliance      : Yes
```

The model provides a risk prediction based on the combination of these vendor performance indicators.

---

# 📁 Project Structure

```text
Vendor-Risk-Scorecard/
│
├── 📄 app.py
├── 📄 requirements.txt
├── 🤖 vendor_risk_model.pkl
├── ⚙️ vendor_risk_scaler.pkl
├── 📋 model_features.json
├── 📊 vendor_ranking.csv
└── 📖 README.md
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming |
| 🐼 Pandas | Data Processing |
| 🔢 NumPy | Numerical Operations |
| 🤖 Scikit-learn | Machine Learning |
| 📈 Logistic Regression | Classification |
| ⚖️ StandardScaler | Feature Scaling |
| 🔍 SHAP | Explainable AI |
| 🌐 Streamlit | Web Application |
| 💾 Joblib | Model Saving |
| 🐙 GitHub | Version Control |
| ☁️ Streamlit Cloud | Deployment |

---

# ▶️ Run the Application Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project directory

```bash
cd Vendor-Risk-Scorecard
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The Vendor Risk Scorecard will open in your web browser.

---

# ☁️ Deployment

The application is designed for deployment using **Streamlit Cloud**.

### Deployment Workflow

```text
GitHub Repository
        ↓
Streamlit Cloud
        ↓
Select Repository
        ↓
Select app.py
        ↓
Install requirements.txt
        ↓
Deploy
        ↓
Live Vendor Risk Scorecard
```

---

# ⚠️ Project Assumptions

The following assumptions were made during the project:

- Defect rate above **5%** represents high defect risk.
- Delivery time above **15 days** represents late-delivery risk.
- `Compliance = No` represents compliance risk.
- Missing defective-unit values were treated as zero after retaining a missing-data indicator.
- Missing delivery days were replaced using the median delivery time.
- A risk score of **2 or more** represents High Risk.

These thresholds are project assumptions and may be modified according to actual procurement policies.

---

# ⚠️ Limitations

The project has the following limitations:

- The dataset contains only **777 purchase-order records**.
- Only **five suppliers** are represented.
- The High Risk target was derived from rule-based risk indicators rather than an independently observed historical supplier-risk outcome.
- Therefore, the model should be considered a **decision-support system**, not a guaranteed predictor of supplier failure.
- A larger historical dataset would improve model reliability.
- Real-world procurement decisions may require additional factors such as:
  - Supplier financial stability
  - Contract performance
  - Geographic risk
  - Production capacity
  - Supplier reputation
  - Business continuity risk

---

# 🚀 Future Enhancements

Future versions of the project can include:

- 📈 Supplier performance trend analysis
- 🤖 Random Forest and Gradient Boosting models
- 🔧 Hyperparameter tuning
- 📊 Model calibration
- 🔄 Automated data updates
- 🚨 Automated supplier alerts
- 🔍 Interactive SHAP explanations inside the dashboard
- 📅 Historical supplier performance tracking
- 💰 Cost-risk optimization
- 🏢 Integration with enterprise procurement systems

---

# 🎓 Academic Value

This project demonstrates a complete end-to-end Machine Learning workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Risk Scoring
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Explainable AI
      ↓
Vendor Ranking
      ↓
Procurement Recommendations
      ↓
Streamlit Application
      ↓
Cloud Deployment
```

The project combines:

**Data Analytics + Machine Learning + Explainable AI + Web Application + Cloud Deployment**

into a practical supplier risk assessment solution.

---

# ⭐ Project Highlights

<p align="center">

| 📊 Dataset | 🤖 Model | 🎯 Accuracy | 🔍 XAI |
|---|---|---|---|
| 777 Records | Logistic Regression | **93.59%** | SHAP |

| 🏢 Suppliers | 🏆 Ranking | 🌐 Application | ☁️ Deployment |
|---|---|---|---|
| 5 | Risk-based | Streamlit | Streamlit Cloud |

</p>

---

# 📌 Conclusion

The **Vendor Risk Scorecard** provides a structured approach for evaluating supplier performance using quality, delivery, compliance, and pricing information.

The Logistic Regression model achieved **93.59% accuracy** on the test dataset.

Vendor-level analysis identified:

- **Delta_Logistics** as the highest-risk supplier.
- **Beta_Supplies** as another high-risk supplier.
- **Gamma_Co** as a medium-risk supplier.
- **Epsilon_Group** and **Alpha_Inc** as relatively low-risk suppliers.

SHAP analysis improved model interpretability by identifying the most influential features.

The Streamlit application provides an interactive interface for vendor risk assessment and procurement decision support.

---

# 👩‍💻 Author

**Vendor Risk Scorecard**

Machine Learning & Data Analytics Academic Project

---

<p align="center">

### ⭐ Vendor Risk Scorecard — From Supplier Data to Intelligent Procurement Decisions

</p>
---

# 📝 Assumptions

The following assumptions were made during the development of the Vendor Risk Scorecard:

1. The historical supplier performance data is assumed to be representative of general vendor performance.

2. A higher defect rate is assumed to indicate a higher level of supplier quality risk.

3. Longer delivery times are assumed to indicate increased delivery and operational risk.

4. A `No` compliance value is assumed to represent higher compliance risk.

5. Price savings are considered as one of the procurement performance indicators.

6. Missing defective-unit values were treated as zero while a separate `Defect_Data_Missing` indicator was retained to preserve information about the original missing values.

7. Missing delivery dates were treated as unavailable delivery information rather than assigning an artificial delivery date.

8. The `High_Risk` target was created using the engineered vendor risk indicators and risk score.

9. The Machine Learning model is intended to support procurement decisions and is not intended to completely replace human judgment.

10. The historical data and model results are assumed to be sufficiently reliable for demonstrating the vendor risk assessment approach.

---

# ⚠️ Limitations

Although the project provides useful vendor risk insights, it has several limitations:

1. The dataset contains only **777 records**, which is relatively small for a production-level Machine Learning system.

2. The dataset contains only **five suppliers**, so the model may not generalize to a large number of suppliers.

3. Some records originally contained missing delivery dates and defective-unit information.

4. Historical supplier performance may not always accurately represent future supplier performance.

5. The model uses only a limited number of risk indicators:
   - Defect Rate
   - Delivery Days
   - Price Savings Percentage
   - Compliance

6. Other important supplier factors such as financial stability, market conditions, transportation disruptions, supplier capacity, contract history, and geopolitical risks are not included.

7. The `High_Risk` target is derived from engineered risk indicators. Therefore, the model learns patterns based on the rules used to create the target.

8. The model should not be used as the only factor when making important procurement or supplier-selection decisions.

9. SHAP explains how features contribute to the model's prediction, but SHAP values do not prove that a feature directly causes supplier risk.

10. The Streamlit application is a demonstration and decision-support system and would require additional validation, monitoring, and larger datasets before being used in a real production environment.

---

# 🎯 Conclusions

The **Vendor Risk Scorecard** successfully demonstrates how Machine Learning and Explainable AI can be used to assess supplier performance and support procurement decision-making.

The project followed the complete workflow from:

```text
Data Collection
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Risk Indicator Creation
      ↓
Feature Scaling
      ↓
Train/Test Split
      ↓
Machine Learning Model
      ↓
Model Evaluation
      ↓
Vendor Risk Ranking
      ↓
Explainable AI using SHAP
      ↓
Streamlit Application
      ↓
Streamlit Cloud Deployment
