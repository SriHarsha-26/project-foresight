"""
Core forecasting logic for FORESIGHT. Streamlit app just imports these two
functions and renders whatever comes back - keeping the model stuff
separate so it's easier to swap the algorithm later without touching the UI.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _build_features(sku_df):
    """
    Turns a single SKU's daily sales history into a feature table the
    model can actually use. Random Forest doesn't understand "time"
    directly so we have to hand it lag values and rolling stats instead.
    """
    sku_df = sku_df.sort_values("Date").reset_index(drop=True)

    # fill any missing sales days with the median for this SKU rather than
    # dropping the row - dropping would break the lag/rolling calculations
    # since they depend on consecutive days
    if sku_df["Units_Sold"].isna().any():
        fill_val = sku_df["Units_Sold"].median()
        if pd.isna(fill_val):
            fill_val = 0
        sku_df["Units_Sold"] = sku_df["Units_Sold"].fillna(fill_val)

    sku_df["lag_1"] = sku_df["Units_Sold"].shift(1)
    sku_df["lag_2"] = sku_df["Units_Sold"].shift(2)
    sku_df["lag_7"] = sku_df["Units_Sold"].shift(7)
    sku_df["rolling_mean_3"] = sku_df["Units_Sold"].shift(1).rolling(window=3).mean()
    sku_df["rolling_mean_7"] = sku_df["Units_Sold"].shift(1).rolling(window=7).mean()
    sku_df["day_of_week"] = pd.to_datetime(sku_df["Date"]).dt.dayofweek

    return sku_df


def train_and_predict(df, sku_id, forecast_days=14):
    """
    Trains a RandomForest on one SKU's history and rolls forward day by
    day to produce a forecast. Returns a DataFrame with Date + Predicted_Units.

    forecast_days: how many days out to predict, defaults to 14 since that's
    roughly the reorder cycle we're planning around
    """
    sku_df = df[df["SKU_ID"] == sku_id].copy()

    if sku_df.empty:
        raise ValueError(f"no rows found for SKU_ID '{sku_id}'")

    sku_df = _build_features(sku_df)

    feature_cols = ["lag_1", "lag_2", "lag_7", "rolling_mean_3", "rolling_mean_7", "day_of_week"]
    train_df = sku_df.dropna(subset=feature_cols)

    # need at least a handful of rows to train on, otherwise the lag_7
    # feature alone eats the first week and there's nothing left
    if len(train_df) < 5:
        # not enough history to build a real model - just fall back to a
        # flat forecast using whatever average we can compute
        avg = sku_df["Units_Sold"].mean()
        avg = 0 if pd.isna(avg) else avg
        last_date = pd.to_datetime(sku_df["Date"]).max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        return pd.DataFrame({
            "Date": future_dates,
            "Predicted_Units": [round(avg, 1)] * forecast_days
        })

    X_train = train_df[feature_cols]
    y_train = train_df["Units_Sold"]

    # 200 trees is overkill for this dataset size but it's fast enough and
    # gives more stable predictions than the default 100
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # roll forward one day at a time, feeding each prediction back in as
    # the next day's lag_1 so the model can "see" its own recent output
    history = sku_df["Units_Sold"].tolist()
    last_date = pd.to_datetime(sku_df["Date"]).max()

    predictions = []
    future_dates = []

    for step in range(forecast_days):
        next_date = last_date + pd.Timedelta(days=step + 1)
        future_dates.append(next_date)

        lag_1 = history[-1] if len(history) >= 1 else 0
        lag_2 = history[-2] if len(history) >= 2 else lag_1
        lag_7 = history[-7] if len(history) >= 7 else lag_1
        roll_3 = np.mean(history[-3:]) if len(history) >= 3 else np.mean(history)
        roll_7 = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)
        dow = next_date.dayofweek

        row = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_7": lag_7,
            "rolling_mean_3": roll_3,
            "rolling_mean_7": roll_7,
            "day_of_week": dow,
        }])

        pred = model.predict(row)[0]
        pred = max(pred, 0)  # can't sell negative units

        predictions.append(round(pred, 1))
        history.append(pred)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted_Units": predictions
    })


def calculate_inventory_metrics(df, predictions, sku_id=None):
    """
    Computes safety stock, reorder point, and a status flag for a SKU
    based on historical sales variability and the forecasted demand.

    df should be the full history (used to get lead time + sales std dev),
    predictions is the output of train_and_predict for the same SKU.
    """
    if sku_id is not None:
        sku_df = df[df["SKU_ID"] == sku_id].copy()
    else:
        sku_df = df.copy()

    if sku_df.empty:
        raise ValueError("no history available to compute inventory metrics")

    sales = sku_df["Units_Sold"].dropna()

    # can't do much with zero or one data point - std dev is undefined/zero
    # in that case, so just treat variability as 0 instead of crashing
    daily_std = sales.std() if len(sales) > 1 else 0
    if pd.isna(daily_std):
        daily_std = 0

    avg_daily_sales = sales.mean() if len(sales) > 0 else 0
    if pd.isna(avg_daily_sales):
        avg_daily_sales = 0

    lead_time = sku_df["Lead_Time_Days"].iloc[-1]
    if pd.isna(lead_time) or lead_time <= 0:
        lead_time = 7  # reasonable fallback if lead time data is missing/bad

    current_stock = sku_df["Current_Stock"].iloc[-1]
    if pd.isna(current_stock):
        current_stock = 0

    # 1.65 multiplier represents a 95% service level factor (z-score for 95% on a
    # one-sided normal distribution) - standard formula, not something we tuned
    safety_stock = 1.65 * daily_std * np.sqrt(lead_time)
    reorder_point = (avg_daily_sales * lead_time) + safety_stock

    forecast_total = predictions["Predicted_Units"].sum() if predictions is not None and len(predictions) > 0 else 0

    # classify based on where current stock sits relative to ROP and the
    # demand we expect to see over the forecast window
    if current_stock <= reorder_point:
        status = "Stockout Risk"
    elif current_stock > (reorder_point + forecast_total):
        status = "Overstock Risk"
    else:
        status = "Optimal"

    return {
        "avg_daily_sales": round(avg_daily_sales, 2),
        "daily_std_dev": round(daily_std, 2),
        "lead_time_days": lead_time,
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "current_stock": current_stock,
        "forecast_total_14d": round(forecast_total, 2),
        "status": status,
    }


if __name__ == "__main__":
    # quick manual smoke test - not part of the module's real interface
    df = pd.read_csv("data/inventory_demand.csv")
    test_sku = df["SKU_ID"].iloc[0]

    preds = train_and_predict(df, test_sku, forecast_days=14)
    print(preds)

    metrics = calculate_inventory_metrics(df, preds, sku_id=test_sku)
    print(metrics)
