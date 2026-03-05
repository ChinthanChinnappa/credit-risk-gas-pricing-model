import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class FICORatingQuantizer:
    def __init__(self, n_buckets):
        self.n_buckets = n_buckets
        self.boundaries = None
        self.method = None

    # -------------------------------------------------
    # 1. Equal Frequency Bucketing (Baseline)
    # -------------------------------------------------
    def fit_equal_frequency(self, fico_scores):
        fico_scores = np.sort(fico_scores)
        quantiles = np.linspace(0, 1, self.n_buckets + 1)
        self.boundaries = np.quantile(fico_scores, quantiles)
        self.method = "equal_frequency"

    # -------------------------------------------------
    # 2. Mean Squared Error Minimization (KMeans)
    # -------------------------------------------------
    def fit_mse(self, fico_scores):
        fico_scores = np.array(fico_scores).reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.n_buckets, random_state=42, n_init=10)
        kmeans.fit(fico_scores)

        centers = np.sort(kmeans.cluster_centers_.flatten())

        # Convert centers to boundaries
        midpoints = (centers[:-1] + centers[1:]) / 2
        self.boundaries = np.concatenate((
            [fico_scores.min()],
            midpoints,
            [fico_scores.max()]
        ))

        self.method = "mse_kmeans"

    # -------------------------------------------------
    # 3. Log-Likelihood Maximization (Dynamic Programming)
    # -------------------------------------------------
    def fit_log_likelihood(self, fico_scores, defaults):
        df = pd.DataFrame({
            "fico": fico_scores,
            "default": defaults
        }).sort_values("fico").reset_index(drop=True)

        n = len(df)
        K = self.n_buckets

        cum_defaults = np.cumsum(df["default"])
        cum_total = np.arange(1, n + 1)

        def bucket_log_likelihood(i, j):
            total = cum_total[j] - (cum_total[i - 1] if i > 0 else 0)
            defaults = cum_defaults[j] - (cum_defaults[i - 1] if i > 0 else 0)

            if total == 0:
                return 0

            p = defaults / total

            if p == 0 or p == 1:
                return 0

            return defaults * np.log(p) + (total - defaults) * np.log(1 - p)

        dp = np.full((K, n), -np.inf)
        split = np.zeros((K, n), dtype=int)

        # Base case: 1 bucket
        for j in range(n):
            dp[0, j] = bucket_log_likelihood(0, j)

        # Fill DP table
        for k in range(1, K):
            for j in range(k, n):
                for i in range(k - 1, j):
                    value = dp[k - 1, i] + bucket_log_likelihood(i + 1, j)
                    if value > dp[k, j]:
                        dp[k, j] = value
                        split[k, j] = i

        # Backtrack
        boundaries_idx = []
        k = K - 1
        j = n - 1

        while k >= 0:
            i = split[k, j]
            boundaries_idx.append(j)
            j = i
            k -= 1

        boundaries_idx = sorted(boundaries_idx)
        self.boundaries = df["fico"].iloc[boundaries_idx].values
        self.method = "log_likelihood_dp"

    # -------------------------------------------------
    # Transform FICO to Rating
    # Lower rating = better credit score
    # -------------------------------------------------
    def transform(self, fico_scores):
        if self.boundaries is None:
            raise ValueError("Model not fitted yet.")

        ratings = np.digitize(fico_scores, self.boundaries, right=True)

        # Invert so lower rating = better score
        ratings = self.n_buckets - ratings + 1

        return ratings


# -------------------------------------------------
# Example Usage
# -------------------------------------------------
if __name__ == "__main__":

    # Replace with your dataset file name
    dataset_path = "customer_loan_data.csv"

    df = pd.read_csv(dataset_path)

    fico_scores = df["fico_score"].values
    defaults = df["default"].values

    quantizer = FICORatingQuantizer(n_buckets=5)

    # Choose ONE of these:
    quantizer.fit_log_likelihood(fico_scores, defaults)
    # quantizer.fit_mse(fico_scores)
    # quantizer.fit_equal_frequency(fico_scores)

    ratings = quantizer.transform(fico_scores)

    df["rating"] = ratings

    print("Method Used:", quantizer.method)
    print("Bucket Boundaries:", quantizer.boundaries)
    print(df[["fico_score", "rating"]].head())