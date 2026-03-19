"""
sql_queries.py — Reusable SQL query functions for the Churn Analysis SQLite DB.
Each function takes db_path as argument and returns a pandas DataFrame.

Interview talking point: Separating SQL logic into a module makes it testable,
reusable, and shows software engineering best practices alongside data science.
"""

import sqlite3
import pandas as pd


def get_high_value_churners(db_path: str) -> pd.DataFrame:
    """
    Returns customers who churned AND had MonthlyCharges > $70.
    These are the highest priority for win-back campaigns.
    """
    query = """
        SELECT
            customerID,
            MonthlyCharges,
            tenure,
            rfm_score,
            Churn
        FROM customers
        WHERE MonthlyCharges > 70
          AND Churn = 'Yes'
        ORDER BY MonthlyCharges DESC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def get_revenue_at_risk_by_segment(db_path: str) -> pd.DataFrame:
    """
    Total monthly revenue at risk per Contract type.
    Revenue at risk = sum of MonthlyCharges for churned customers.

    Interview talking point: This query directly answers the CFO question —
    'Where is our biggest revenue risk by segment?'
    """
    query = """
        SELECT
            CASE
                WHEN "Contract_Month-to-month" = 1 THEN 'Month-to-month'
                WHEN "Contract_One year"        = 1 THEN 'One year'
                WHEN "Contract_Two year"        = 1 THEN 'Two year'
                ELSE 'Unknown'
            END AS contract_type,
            COUNT(*)                    AS churned_customers,
            ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue_lost,
            ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge
        FROM customers
        WHERE Churn = 'Yes'
        GROUP BY contract_type
        ORDER BY total_monthly_revenue_lost DESC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def get_rfm_churn_summary(db_path: str) -> pd.DataFrame:
    """
    Churn rate and average charges per RFM score group.
    Shows whether higher RFM score customers churn less (validates the feature).
    """
    query = """
        SELECT
            rfm_score,
            COUNT(*)                          AS customer_count,
            ROUND(AVG(MonthlyCharges), 2)     AS avg_monthly_charges,
            ROUND(AVG(Churn_Binary) * 100, 1) AS churn_rate_pct
        FROM customers
        GROUP BY rfm_score
        ORDER BY rfm_score ASC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
