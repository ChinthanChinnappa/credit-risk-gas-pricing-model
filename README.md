# Credit Risk Modeling System

### Probability of Default (PD) & FICO Rating Quantization

This project implements a **credit risk modeling pipeline** used in financial institutions to evaluate loan risk. It includes:

1. Probability of Default (PD) prediction
2. Expected Loss estimation
3. FICO score rating bucket optimization

The system demonstrates how banks convert **raw borrower data into risk metrics used for lending decisions**.


# Project Architecture

```
credit-risk-model/
│
├── loan_pd_expected_loss_model.py
├── fico_rating_quantization.py
├── customer_loan_data.csv
└── README.md
```

---

# Key Features

## 1. Probability of Default (PD) Model

Implemented using **machine learning models**:

* Logistic Regression
* Random Forest

The system automatically selects the **best performing model based on ROC-AUC score**.

Outputs:

* Probability of Default (PD)
* Expected Loss (EL)

Expected Loss formula:

```
Expected Loss = PD × Loss Given Default × Exposure
```

---

## 2. Expected Loss Estimation

The model estimates potential financial losses for lenders.

Inputs:

* Borrower features
* Loan exposure amount
* Recovery rate

Example output:

```
Predicted PD: 0.083
Expected Loss: $7470
```

---

## 3. FICO Rating Quantization

This module converts continuous **FICO credit scores into rating buckets**.

Three different quantization techniques are implemented:

### 1. Equal Frequency Bucketing

Splits scores so each bucket has similar number of borrowers.

### 2. Mean Squared Error Minimization

Uses **K-Means clustering** to group similar credit scores.

### 3. Log-Likelihood Maximization (Advanced)

Uses **dynamic programming** to create rating buckets that best explain default probabilities.

This technique is used in **banking risk modeling and Basel frameworks**.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Machine Learning
* Credit Risk Modeling

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/credit-risk-model.git
cd credit-risk-model
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

# Usage

## Train Probability of Default Model

```python
from loan_pd_expected_loss_model import LoanPDModel

model = LoanPDModel()
model.train("customer_loan_data.csv")
```

---

## Predict Borrower Risk

```python
borrower = {
    "income": 50000,
    "total_loans_outstanding": 20000,
    "credit_score": 650,
    "years_employed": 5
}

pd = model.predict_pd(borrower)
expected_loss = model.expected_loss(borrower, exposure=100000)
```

---

## Generate FICO Rating Buckets

```python
from fico_rating_quantization import FICORatingQuantizer

quantizer = FICORatingQuantizer(n_buckets=5)

quantizer.fit_log_likelihood(fico_scores, defaults)

ratings = quantizer.transform(fico_scores)
```

---

# Example Dataset

The dataset should contain fields like:

```
income
total_loans_outstanding
credit_score
years_employed
default
fico_score
```

Where:

```
default = 1 → borrower defaulted
default = 0 → borrower did not default
```

---

# Real World Applications

This system reflects techniques used in:

* Investment banks
* Credit risk departments
* Fintech lending platforms
* Basel risk compliance systems

Applications include:

* Loan approval systems
* Credit scoring engines
* Risk-adjusted pricing
* Portfolio risk analysis

---

# Future Improvements

Possible extensions:

* Gradient Boosting models (XGBoost)
* Feature engineering pipelines
* Model explainability (SHAP)
* Credit portfolio simulation
* REST API deployment
* Real-time scoring system

---

# Author

Chinthan Chinnappa P T 

Computer Science Engineering Student
Interested in:

* Data Science
* Quantitative Finance
* Risk Modeling
* Machine Learning Systems
* Backend Systems 

---

# License

This project is open-source and available under the MIT License.

#
