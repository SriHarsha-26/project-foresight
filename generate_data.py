"""
Builds a fake sales/inventory dataset so we have something to build the
forecasting logic against before we get access to the real warehouse feed.

Run this once to spit out data/inventory_demand.csv, then forecaster.py
picks it up from there.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)  # keeping this fixed so the demo numbers don't jump around every run

NUM_DAYS = 50
START_DATE = "2024-01-01"

# made up but roughly realistic - a mix of electronics/grocery/apparel type items
SKU_CATALOG = [
    {"sku": "SKU-1001", "category": "Electronics", "base_demand": 18, "price": 249.99},
    {"sku": "SKU-1002", "category": "Electronics", "base_demand": 9,  "price": 599.00},
    {"sku": "SKU-1003", "category": "Grocery",     "base_demand": 45, "price": 4.50},
    {"sku": "SKU-1004", "category": "Grocery",     "base_demand": 60, "price": 2.25},
    {"sku": "SKU-1005", "category": "Apparel",      "base_demand": 22, "price": 34.99},
    {"sku": "SKU-1006", "category": "Apparel",      "base_demand": 15, "price": 79.99},
    {"sku": "SKU-1007", "category": "Home",         "base_demand": 12, "price": 89.50},
    {"sku": "SKU-1008", "category": "Home",         "base_demand": 8,  "price": 129.00},
    {"sku": "SKU-1009", "category": "Toys",         "base_demand": 20, "price": 19.99},
    {"sku": "SKU-1010", "category": "Toys",         "base_demand": 14, "price": 24.99},
]

LEAD_TIME_BY_CATEGORY = {
    "Electronics": 10,
    "Grocery": 3,
    "Apparel": 7,
    "Home": 12,
    "Toys": 6,
}


def make_demand_curve(base_demand, num_days):
    """
    Builds a demand series with a slow upward trend + weekly seasonality
    (weekends spike) + random noise. Not meant to be super precise, just
    needs to look like real sales data instead of a flat line.
    """
    days = np.arange(num_days)

    # small linear trend - pretend the SKU is slowly gaining popularity
    trend = base_demand + days * (base_demand * 0.01)

    # weekly cycle, day 5/6 (Sat/Sun) get a bump since START_DATE is a Monday
    weekday_pattern = np.array([1.0, 0.95, 1.0, 1.05, 1.15, 1.35, 1.25])
    seasonal_multiplier = weekday_pattern[days % 7]

    noisy = trend * seasonal_multiplier + np.random.normal(0, base_demand * 0.15, size=num_days)

    # units sold can't be negative, and it's a count so round it
    noisy = np.clip(noisy, 0, None)
    return np.round(noisy).astype(int)


def build_dataset():
    dates = pd.date_range(start=START_DATE, periods=NUM_DAYS, freq="D")

    rows = []
    for item in SKU_CATALOG:
        demand_series = make_demand_curve(item["base_demand"], NUM_DAYS)
        lead_time = LEAD_TIME_BY_CATEGORY[item["category"]]

        # start stock somewhere reasonable above what we'll sell in the lead time window
        stock_on_hand = int(item["base_demand"] * lead_time * 1.8)

        for i, date in enumerate(dates):
            units_sold = demand_series[i]
            stock_on_hand = max(stock_on_hand - units_sold, 0)

            # simulate a restock kicking in roughly once a week so stock doesn't hit zero and stay there
            if i % 7 == 6:
                stock_on_hand += int(item["base_demand"] * lead_time * 1.5)

            # unit price drifts a tiny bit day to day, not exactly the same every time
            price_jitter = item["price"] * np.random.uniform(-0.02, 0.02)

            rows.append({
                "Date": date.strftime("%Y-%m-%d"),
                "SKU_ID": item["sku"],
                "Category": item["category"],
                "Units_Sold": units_sold,
                "Unit_Price": round(item["price"] + price_jitter, 2),
                "Current_Stock": stock_on_hand,
                "Lead_Time_Days": lead_time,
            })

    df = pd.DataFrame(rows)

    # sprinkle in a handful of missing values on purpose - real POS exports are never clean
    missing_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[missing_idx, "Units_Sold"] = np.nan

    return df


if __name__ == "__main__":
    df = build_dataset()

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inventory_demand.csv")
    df.to_csv(out_path, index=False)

    print(f"wrote {len(df)} rows to {out_path}")
    print(f"SKUs: {df['SKU_ID'].nunique()}, days: {df['Date'].nunique()}")
