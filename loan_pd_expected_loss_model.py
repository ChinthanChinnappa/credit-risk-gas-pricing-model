import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


class LoanPDModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def train(self, data_path, target_column="default"):
        """
        Train the PD model using provided dataset.

        data_path: path to CSV file
        target_column: column indicating default (1 = default, 0 = no default)
        """

        df = pd.read_csv(data_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.feature_columns = X.columns

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Logistic Regression
        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(X_train, y_train)
        log_pred = log_model.predict_proba(X_test)[:, 1]
        log_auc = roc_auc_score(y_test, log_pred)

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred)

        # Select better model
        if rf_auc > log_auc:
            self.model = rf_model
            print(f"Selected Model: Random Forest | AUC: {rf_auc:.4f}")
        else:
            self.model = log_model
            print(f"Selected Model: Logistic Regression | AUC: {log_auc:.4f}")

    def predict_pd(self, borrower_features):
        """
        borrower_features: dictionary of feature values
        Returns Probability of Default (PD)
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        input_df = pd.DataFrame([borrower_features])

        # Ensure same feature order
        input_df = input_df[self.feature_columns]

        input_scaled = self.scaler.transform(input_df)

        pd_value = self.model.predict_proba(input_scaled)[0][1]

        return pd_value

    def expected_loss(self, borrower_features, exposure, recovery_rate=0.10):
        """
        exposure: loan amount
        recovery_rate: assumed recovery rate (default 10%)
        """

        pd_value = self.predict_pd(borrower_features)
        loss_given_default = 1 - recovery_rate

        expected_loss = pd_value * loss_given_default * exposure

        return expected_loss


if __name__ == "__main__":

    model = LoanPDModel()

    # Replace with your actual dataset filename
    dataset_path = "customer_loan_data.csv"

    model.train(dataset_path, target_column="default")

    sample_borrower = {
        "income": 50000,
        "total_loans_outstanding": 20000,
        "credit_score": 650,
        "years_employed": 5
    }

    exposure_amount = 100000

    pd_value = model.predict_pd(sample_borrower)
    el_value = model.expected_loss(sample_borrower, exposure_amount)

    print(f"Predicted PD: {pd_value:.4f}")
    print(f"Expected Loss: {el_value:.2f}")