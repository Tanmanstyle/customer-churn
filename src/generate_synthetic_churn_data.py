"""
Generate a synthetic fintech/banking customer churn dataset.

This dataset is synthetic, portfolio-safe, and designed to behave like a
realistic customer churn dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(40, 12, n_customers).clip(18, 80).round().astype(int)
    tenure_years = rng.integers(0, 11, n_customers)
    account_balance = rng.lognormal(mean=9.5, sigma=0.75, size=n_customers).round(2)
    credit_score = rng.normal(650, 90, n_customers).clip(300, 850).round().astype(int)
    num_products = rng.choice([1, 2, 3, 4], size=n_customers, p=[0.38, 0.42, 0.15, 0.05])
    is_active = rng.choice([0, 1], size=n_customers, p=[0.35, 0.65])
    estimated_salary = rng.normal(55000, 22000, n_customers).clip(12000, 160000).round(2)
    complaints = rng.poisson(0.35, n_customers).clip(0, 5)
    support_tickets = rng.poisson(1.0, n_customers).clip(0, 8)
    mobile_logins_30d = rng.poisson(12, n_customers).clip(0, 60)
    overdraft_count_12m = rng.poisson(0.8, n_customers).clip(0, 12)

    z = (
        -1.5
        + 0.04 * (age - 40)
        - 0.18 * tenure_years
        - 0.003 * (credit_score - 650)
        + 0.55 * (num_products == 1)
        - 0.85 * is_active
        + 0.65 * complaints
        + 0.18 * support_tickets
        - 0.035 * mobile_logins_30d
        + 0.18 * overdraft_count_12m
        + 0.000002 * account_balance
    )

    churn_probability = sigmoid(z)
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)],
        "age": age,
        "tenure_years": tenure_years,
        "account_balance": account_balance,
        "credit_score": credit_score,
        "num_products": num_products,
        "is_active": is_active,
        "estimated_salary": estimated_salary,
        "complaints": complaints,
        "support_tickets": support_tickets,
        "mobile_logins_30d": mobile_logins_30d,
        "overdraft_count_12m": overdraft_count_12m,
        "churn": churn,
    })

    return df


def main() -> None:
    df = generate_data()
    output_path = DATA_DIR / "fintech_churn_synthetic.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic churn dataset to {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Churn rate: {df['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
