# Black Friday EDA & Customer Segmentation

An analysis of 100k Black Friday transaction records (Nov 24 – Dec 1, 2025). This project includes a Jupyter notebook for exploratory data analysis (EDA), customer re-segmentation using K-Means clustering, and a dark-themed Streamlit dashboard.

## Key Insights
* **Synthetic Dataset Warning:** The pre-existing customer segment labels (`Loyal`, `VIP`, `New`, `Returning`) are completely randomized/synthetic. They all share identical average Recency, Frequency, and Monetary (RFM) scores.
* **Electronics Dominance:** Electronics was the largest revenue driver, contributing **$14.12M (40.2%)** out of the total **$35.13M** sales. Home & Kitchen was second at 18.4%.
* **Optimal Discounts:** 20%-30% discounts drove the highest revenue. Extreme discounts (50%-60%) showed a massive drop in both sales volume and revenue.
* **K-Means RFM Segments:** Re-clustering the customers using actual RFM metrics successfully mapped 4 distinct tiers:
  * **Champions/VIPs** (7.4%): High spend ($4.2k avg), high frequency.
  * **Loyal Shoppers** (29.9%): Consistent purchases ($1.3k avg spend).
  * **Occasional Shoppers** (36.3%): Active but lower spend ($600 avg).
  * **At-Risk** (26.4%): Dormant customers with low spend (~$547 avg).

## File Structure
* `main.ipynb`: Data quality checks, EDA, and K-Means RFM clustering implementation.
* `app.py`: Streamlit dashboard with interactive filters (gender, city, age group, category, payment method) and a 3D cluster plot.
* `Dataset_Black_Friday_Explanation.md`: Table schema and column descriptions.
* `plots/`: Static charts exported from the notebook analysis.

## Setup & Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
