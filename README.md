### Credit Risk Model Development Report: 
##  Probability of Default (PD) Modeling

This repository contains the analysis, code, and interpretation for developing a Logistic Regression model to predict the Probability of Default (PD) for loan applicants, addressing the challenge of severe class imbalance.

---

### Data Understanding and Cleaning 

#### Summary and Target Definition
* **Observations:** 2,500 loan records.
* **Target Variable (`Default`):** Created from `loan_status` where ('default' and '90+dpd') are classified as **1 (Default)**.
* **Class Imbalance:** The Default Rate was **20.84%** (approx. 4:1 Non-Default to Default ratio), which became the focus of the modeling strategy.

#### Data Cleaning
* **Missing Values:** Found in `income`, `loan_amount`, and `loan_purpose`. Numerical columns were imputed using the **median** value.
* **Outliers:** Outliers in financial metrics were **retained** as they are critical information for accurate risk profiling and were handled by subsequent scaling.

---

### Exploratory Data Analysis 

#### Bivariate Analysis Highlights

| Feature | Observation |
| :--- | :--- |
| **Credit Score** | Defaulters had a slightly higher average score than non-defaulters, signaling a **non-linear or masked relationship**. |
| **Loan Term** | 60-month loans ($\text{PD} \approx 22.1\%$) were consistently riskier than 36-month loans ($\text{PD} \approx 19.6\%$). |
| **Loan Purpose** | Loans for **Education** and **Car** showed the highest default rates, proving categorical features are strong differentiators. |

#### Visual Interpretations 

 **Income Distribution by Default Status:** 
 ![Income Distribution by Default Status](images\vis-1.png)

    * *Interpretation:* The heavy overlap between the two classes confirmed that raw income is a weak linear predictor, justifying the creation of the DTI ratio.


 **Correlation Heatmap:** t
 ![Correlation Heatmap](images\vis-2.png)
    * *Interpretation:* Showed near-zero correlation between raw numerical features and the `Default` target, reinforcing the need for engineered features.

**Default Rate by Loan Purpose** 
![Default Rate by Loan Purpose](images\vis-3.png)
| Clearly illustrated the varying default rates across categories. 
---

### Feature Engineering 

Two features were engineered to capture relationships missed by raw data:

**Debt-to-Income Ratio (`DTI_Ratio`):** Calculated as $\frac{\text{loan\_amount}}{\text{income}}$.
    * **Justification:** A core risk metric that directly measures **debt serviceability**.

**Age-Risk Group (`Age_Group`):** Age was binned to capture the known **non-linear risk curve** associated with age.

**Final Data Prep:** All numerical features were transformed using **Standard Scaling**, and categorical features were converted using **One-Hot Encoding**.

---

### Probability of Default (PD) Modeling 

#### Baseline Model Failure (Unweighted Logistic Regression)

The initial model, fitted without addressing the 4:1 class imbalance, failed.

| Metric | Baseline Result | Interpretation |
| :--- | :--- | :--- |
| **Confusion Matrix** | $\begin{bmatrix} 396 & 0 \\ 104 & 0 \end{bmatrix}$ | **Critical Failure: Zero True Positives (TP=0)**. The model only predicted the majority class (Non-Default). |
| **AUC** | 0.4896 | Worse than random guessing. |

#### Coefficient Interpretation (Baseline)
| Feature | Odds Ratio ($e^\beta$) | Finding |
| :--- | :--- | :--- |
| **`DTI_Ratio`** | $\mathbf{1.871}$ | **Strongest Predictor:** A one-unit increase in DTI increases the odds of default by **87.1%**. (Intuitive) |
| **`credit_score`** | $1.128$ | **Counter-Intuitive:** An increase in credit score increases the odds of default, signaling model instability due to imbalance. |

---

### Model Improvement and Recommendations 

#### Weakness and Recommendation
The primary recommendation was to address the fatal flaw of **class imbalance**. This was implemented by running a **Weighted Logistic Regression** using the $\text{class\_weight}=\text{'balanced'}$ parameter.

#### Weighted Model Evaluation (Final Result)

| Metric | Weighted LogReg Result | Improvement from Baseline |
| :--- | :--- | :--- |
| **AUC** | $\mathbf{0.6866}$ | Major gain in discriminative power. |
| **KS Statistic** | $\mathbf{0.2520}$ | Acceptable separation of risk scores. |
| **Confusion Matrix** | $\begin{bmatrix} 280 & 116 \\ 34 & 70 \end{bmatrix}$ | **Success:** $\mathbf{70}$ True Positives were correctly identified, stabilizing the model. |
![Weighted Logistic Regression Roc Curve](images\vis-4.png)

This final weighted model provides a robust, interpretable, and reasonably predictive baseline for assessing the Probability of Default.