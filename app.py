import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from forecaster import train_and_predict, calculate_inventory_metrics

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# dark-ish accent palette reused across all the charts so everything matches
COLOR_HISTORY = "#5B8DEF"
COLOR_FORECAST = "#F2A93B"
COLOR_STOCK = "#3BB273"
COLOR_DEMAND = "#E85D75"
STATUS_COLORS = {
    "Stockout Risk": "#E85D75",
    "Overstock Risk": "#F2A93B",
    "Optimal": "#3BB273",
}

FORECAST_DAYS = 14


@st.cache_data
def load_data(path="data/inventory_demand.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


@st.cache_data(show_spinner=False)
def build_forecasts(df, sku_list, forecast_days=FORECAST_DAYS):
    """
    Runs train_and_predict for every SKU we care about and hands back a
    dict of sku -> forecast dataframe. Wrapped in cache_data since the
    RandomForest retrain is the slowest part of the app by a wide margin.
    """
    results = {}
    for sku in sku_list:
        try:
            results[sku] = train_and_predict(df, sku, forecast_days=forecast_days)
        except ValueError:
            # sku has no rows in this date range, just skip it
            continue
    return results


@st.cache_data(show_spinner=False)
def build_metrics(df, sku_list, _forecasts):
    """
    Runs calculate_inventory_metrics per SKU. Leading underscore on
    _forecasts tells streamlit not to try to hash the dict of dataframes,
    we key the cache off df + sku_list instead.
    """
    rows = []
    for sku in sku_list:
        if sku not in _forecasts:
            continue
        metrics = calculate_inventory_metrics(df, _forecasts[sku], sku_id=sku)
        metrics["SKU_ID"] = sku
        metrics["Category"] = df.loc[df["SKU_ID"] == sku, "Category"].iloc[-1]
        rows.append(metrics)
    return pd.DataFrame(rows)


def recommend_action(row):
    """Turns a status flag into something a planner can actually act on."""
    if row["status"] == "Stockout Risk":
        reorder_qty = max(row["reorder_point"] + row["forecast_total_14d"] - row["current_stock"], 0)
        return pd.Series({
            "Recommended_Action": "Reorder Now",
            "Suggested_Qty": round(reorder_qty),
            "Priority": "High",
        })
    elif row["status"] == "Overstock Risk":
        return pd.Series({
            "Recommended_Action": "Hold / Consider Promotion",
            "Suggested_Qty": 0,
            "Priority": "Low",
        })
    else:
        return pd.Series({
            "Recommended_Action": "Monitor",
            "Suggested_Qty": 0,
            "Priority": "Medium",
        })


def style_status_col(val):
    color = STATUS_COLORS.get(val, "#888888")
    return f"background-color: {color}20; color: {color}; font-weight: 600"


def style_priority_col(val):
    priority_colors = {"High": "#E85D75", "Medium": "#F2A93B", "Low": "#3BB273"}
    color = priority_colors.get(val, "#888888")
    return f"background-color: {color}20; color: {color}; font-weight: 600"


df_full = load_data()

st.title("📦 Project FORESIGHT")
st.caption("AI-Driven Demand Forecasting & Inventory Intelligence")

# ---------- sidebar ----------
st.sidebar.header("Filters")

all_skus = sorted(df_full["SKU_ID"].unique())
sku_choice = st.sidebar.selectbox("SKU", ["All SKUs"] + all_skus)

min_date = df_full["Date"].min().date()
max_date = df_full["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# date_input gives back a single date until the user picks the second one,
# so guard against that instead of crashing on unpacking
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)
df = df_full.loc[mask].copy()

if df.empty:
    st.warning("No data in the selected date range. Widen the filter to see results.")
    st.stop()

selected_skus = all_skus if sku_choice == "All SKUs" else [sku_choice]

with st.spinner("Training forecast models..."):
    forecasts = build_forecasts(df, selected_skus)
    metrics_df = build_metrics(df, selected_skus, forecasts)

if metrics_df.empty:
    st.warning("Not enough data to compute forecasts for this selection.")
    st.stop()

# ---------- top metric cards ----------
latest_per_sku = df.sort_values("Date").groupby("SKU_ID").tail(1)
latest_per_sku = latest_per_sku[latest_per_sku["SKU_ID"].isin(selected_skus)]

total_inventory_value = (latest_per_sku["Current_Stock"] * latest_per_sku["Unit_Price"]).sum()
at_risk_count = metrics_df["status"].isin(["Stockout Risk", "Overstock Risk"]).sum()
total_forecast_demand = metrics_df["forecast_total_14d"].sum()

# revenue at risk = units we'd fall short on for stockout SKUs, priced at their unit price
stockout_rows = metrics_df[metrics_df["status"] == "Stockout Risk"].copy()
price_lookup = latest_per_sku.set_index("SKU_ID")["Unit_Price"]
stockout_rows["shortfall_units"] = (
    stockout_rows["forecast_total_14d"] - stockout_rows["current_stock"]
).clip(lower=0)
stockout_rows["unit_price"] = stockout_rows["SKU_ID"].map(price_lookup).fillna(0)
revenue_at_risk = (stockout_rows["shortfall_units"] * stockout_rows["unit_price"]).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Inventory Value", f"${total_inventory_value:,.0f}")
c2.metric("At-Risk SKUs", f"{at_risk_count} / {len(metrics_df)}")
c3.metric("Forecasted 14-Day Demand", f"{total_forecast_demand:,.0f} units")
c4.metric("Potential Revenue at Risk", f"${revenue_at_risk:,.0f}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "📈 Demand Forecasting",
    "⚠️ Inventory Risk & ROP",
    "✅ Action Plan & Export",
])

# ---------- tab 1: forecasting ----------
with tab1:
    st.subheader("Historical vs Forecasted Demand")

    if sku_choice == "All SKUs":
        history = df.groupby("Date", as_index=False)["Units_Sold"].sum()
        forecast_frames = [f.assign(SKU_ID=sku) for sku, f in forecasts.items()]
        forecast = pd.concat(forecast_frames).groupby("Date", as_index=False)["Predicted_Units"].sum()
        chart_title = "All SKUs — Combined Daily Units Sold"
    else:
        history = df[df["SKU_ID"] == sku_choice][["Date", "Units_Sold"]]
        forecast = forecasts.get(sku_choice, pd.DataFrame(columns=["Date", "Predicted_Units"]))
        chart_title = f"{sku_choice} — Daily Units Sold"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["Date"], y=history["Units_Sold"],
        mode="lines", name="Historical Sales",
        line=dict(color=COLOR_HISTORY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["Predicted_Units"],
        mode="lines+markers", name="14-Day Forecast",
        line=dict(color=COLOR_FORECAST, width=2, dash="dash"),
        marker=dict(size=5),
    ))

    fig.update_layout(
        template="plotly_dark",
        title=chart_title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Units",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig, width='stretch')

# ---------- tab 2: risk & rop ----------
with tab2:
    st.subheader("Current Stock vs Forecasted Demand")

    bar_df = metrics_df[["SKU_ID", "current_stock", "forecast_total_14d"]].copy()
    bar_df = bar_df.sort_values("current_stock", ascending=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bar_df["SKU_ID"], y=bar_df["current_stock"],
        name="Current Stock", marker_color=COLOR_STOCK,
    ))
    fig2.add_trace(go.Bar(
        x=bar_df["SKU_ID"], y=bar_df["forecast_total_14d"],
        name="Forecasted 14-Day Demand", marker_color=COLOR_DEMAND,
    ))

    fig2.update_layout(
        template="plotly_dark",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="SKU",
        yaxis_title="Units",
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

    st.plotly_chart(fig2, width='stretch')

    st.subheader("Reorder Point Detail")

    rop_table = metrics_df[[
        "SKU_ID", "Category", "avg_daily_sales", "daily_std_dev", "lead_time_days",
        "safety_stock", "reorder_point", "current_stock", "status",
    ]].rename(columns={
        "avg_daily_sales": "Avg Daily Sales",
        "daily_std_dev": "Daily Std Dev",
        "lead_time_days": "Lead Time (days)",
        "safety_stock": "Safety Stock",
        "reorder_point": "Reorder Point",
        "current_stock": "Current Stock",
        "status": "Status",
    })

    styled_rop = rop_table.style.map(style_status_col, subset=["Status"])
    st.dataframe(styled_rop, width='stretch', hide_index=True)

# ---------- tab 3: action plan ----------
with tab3:
    st.subheader("Recommended Actions")

    action_df = metrics_df.copy()
    action_df = pd.concat([action_df, action_df.apply(recommend_action, axis=1)], axis=1)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    action_df["_sort"] = action_df["Priority"].map(priority_order)
    action_df = action_df.sort_values("_sort").drop(columns="_sort")

    display_cols = [
        "SKU_ID", "Category", "status", "current_stock", "reorder_point",
        "forecast_total_14d", "Recommended_Action", "Suggested_Qty", "Priority",
    ]
    action_display = action_df[display_cols].rename(columns={
        "status": "Status",
        "current_stock": "Current Stock",
        "reorder_point": "Reorder Point",
        "forecast_total_14d": "Forecasted Demand (14d)",
    })

    styled_action = (
        action_display.style
        .map(style_status_col, subset=["Status"])
        .map(style_priority_col, subset=["Priority"])
    )
    st.dataframe(styled_action, width='stretch', hide_index=True)

    st.download_button(
        label="⬇️ Export Action Plan as CSV",
        data=action_display.to_csv(index=False).encode("utf-8"),
        file_name="foresight_action_plan.csv",
        mime="text/csv",
    )
