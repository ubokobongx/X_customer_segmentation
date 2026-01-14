import pandas as pd
from sqlalchemy import text
from redshiftlogin import RedshiftConnector

class DataExporter:
    def __init__(self):
        self.db_connector = RedshiftConnector()
        self.engine = self.db_connector.get_engine()
        self.sql_query = """
  WITH recent_loans AS (
    SELECT a.*
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss" a
    JOIN (
        SELECT dw_customer_key, MAX(disbursement_date) AS max_disbursement_date
        FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss"
        GROUP BY dw_customer_key
    ) b
      ON a.dw_customer_key = b.dw_customer_key
     AND a.disbursement_date = b.max_disbursement_date
    WHERE a.product_code NOT IN ('DTP', 'ON2D')
),

loan_counts AS (
    SELECT dw_customer_key, COUNT(DISTINCT loan_ref) AS loan_count
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss"
    WHERE product_code NOT IN ('DTP', 'ON2D')
    GROUP BY dw_customer_key
),

-- Get the LATEST dw_date_key for EACH CUSTOMER (across all their loans)
latest_dates AS (
    SELECT 
        dw_customer_key,
        MAX(dw_date_key) as latest_date_key
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss"
    WHERE product_code NOT IN ('DTP', 'ON2D')
    GROUP BY dw_customer_key
),

-- Get maturity_dpd from the LATEST date for EACH CUSTOMER
-- If multiple loans on same latest date, take the HIGHEST maturity_dpd
latest_maturity_dpd AS (
    SELECT 
        f.dw_customer_key,
        MAX(f.maturity_dpd) as maturity_dpd
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss" f
    INNER JOIN latest_dates l 
        ON f.dw_customer_key = l.dw_customer_key 
        AND f.dw_date_key = l.latest_date_key
    WHERE f.product_code NOT IN ('DTP', 'ON2D')
    GROUP BY f.dw_customer_key
),

-- Check if ANY installment_dpd >= 14 at ANY point for ANY loan
-- AND COUNT how many times it happened
installment_dpd_check AS (
    SELECT 
        dw_customer_key,
        MAX(CASE 
            WHEN installment_dpd >= 14 THEN 1
            ELSE 0
        END) as has_14plus_dpd,
        SUM(CASE 
            WHEN installment_dpd >= 14 THEN 1
            ELSE 0
        END) as count_14plus_dpd
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss"
    WHERE product_code NOT IN ('DTP', 'ON2D')
    GROUP BY dw_customer_key
),

-- For other metrics, let's aggregate across ALL customer loans (not just recent)
customer_aggregates AS (
    SELECT
        dw_customer_key,
        SUM(total_amount_overdue) AS total_amount_overdue,
        SUM(total_loan_amount) AS total_loan_amount,
        SUM(total_missed_installment) AS total_missed_installment,
        AVG(tenor_in_months) AS tenor_in_months,
        AVG(ontime_repayment_rate) AS ontime_repayment_rate
    FROM "ox_datawarehouse"."stg_oxygenx"."fact_loan_details_mss"
    WHERE product_code NOT IN ('DTP', 'ON2D')
    GROUP BY dw_customer_key
),

aggregated_metrics AS (
    SELECT
        r.dw_customer_key,
        c.loan_count,
        ca.total_amount_overdue,
        ca.total_loan_amount,
        COALESCE(l.maturity_dpd, 0) AS maturity_dpd,
        ca.total_missed_installment,
        COALESCE(i.has_14plus_dpd, 0) AS has_14plus_dpd,
        COALESCE(i.count_14plus_dpd, 0) AS count_14plus_dpd,
        ca.tenor_in_months,
        ca.ontime_repayment_rate
    FROM recent_loans r
    LEFT JOIN loan_counts c ON r.dw_customer_key = c.dw_customer_key
    LEFT JOIN latest_maturity_dpd l ON r.dw_customer_key = l.dw_customer_key
    LEFT JOIN installment_dpd_check i ON r.dw_customer_key = i.dw_customer_key
    LEFT JOIN customer_aggregates ca ON r.dw_customer_key = ca.dw_customer_key
    GROUP BY 
        r.dw_customer_key, 
        c.loan_count, 
        l.maturity_dpd,
        i.has_14plus_dpd,
        i.count_14plus_dpd,
        ca.total_amount_overdue,
        ca.total_loan_amount,
        ca.total_missed_installment,
        ca.tenor_in_months,
        ca.ontime_repayment_rate
),

final_data AS (
    SELECT 
        a.dw_customer_key AS customer_id,
        a.loan_count,
        a.total_amount_overdue,
        a.maturity_dpd,
        a.total_missed_installment,
        a.has_14plus_dpd,
        a.count_14plus_dpd,
        a.tenor_in_months,
        a.ontime_repayment_rate,
        d.age,
        d.gender,
        d.marital_status,
        d.state,
        d.location,
        d.purpose_of_loan AS purpose,
        d.employment_status,
        d.dw_channel_key,
        a.total_loan_amount,
        (
            SELECT COALESCE(
                       MAX(CASE WHEN LOWER(td.offer_type) LIKE '%salar%'    THEN td.monthly_salary_l6m END),
                       MAX(CASE WHEN LOWER(td.offer_type) LIKE '%turnover%' THEN td.turnover_last6m_trim END)
                   )
            FROM "ox_datawarehouse"."stg_oxygenx"."taktile_decision" td
            WHERE td.user_client_id = a.dw_customer_key
        ) AS income
    FROM aggregated_metrics a
    INNER JOIN "ox_datawarehouse"."stg_oxygenx"."dim_customer" d
        ON a.dw_customer_key = d.dw_customer_key
    WHERE d.customer_category = 'Individual'
)

SELECT 
    customer_id,
    loan_count,
    total_amount_overdue,
    maturity_dpd,
    total_missed_installment,
    has_14plus_dpd,
    count_14plus_dpd,
    tenor_in_months,
    ontime_repayment_rate,
    age,
    gender,
    marital_status,
    state,
    location,
    purpose,
    employment_status,
    dw_channel_key,
    total_loan_amount,
    income
FROM final_data
WHERE 
    (CASE 
         WHEN income IS NULL OR total_loan_amount IS NULL OR total_loan_amount = 0 THEN NULL
         ELSE ABS(income / total_loan_amount)
     END) <= 100
ORDER BY customer_id;
        """

    def export_to_csv(self, output_file='cust_data.csv'):
        try:
            # Version-compatible approach
            with self.engine.begin() as conn:
                # Method 1: Try SQLAlchemy execution first
                try:
                    df = pd.read_sql_query(text(self.sql_query), conn)
                except TypeError:
                    # Fallback for older pandas versions
                    df = pd.read_sql_query(self.sql_query, conn.connection)
                
                df.to_csv(output_file, index=False)
                print(f"✅ Exported {len(df)} rows to '{output_file}'")
        except Exception as e:
            print(f"❌ Query or export failed: {str(e)}")
            raise