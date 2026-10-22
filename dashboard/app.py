"""RetailPulse - Customer Analytics & Demand Forecasting Dashboard.

Run from the project root:
    streamlit run dashboard/app.py

The dashboard reads the saved outputs in data/processed/ and reports/. The churn
model in models/ is loaded for live scoring; if it cannot be loaded, the
dashboard keeps working with the saved predictions and shows a clear message.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Page setup

st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

SEGMENT_COLORS = {
    "Champions": "#2E7D32",
    "Loyal High-Value Customers": "#1E88E5",
    "Regular Customers": "#8E24AA",
    "Occasional Customers": "#FB8C00",
    "At-Risk Valuable Customers": "#E53935",
    "Inactive Customers": "#757575",
}
RISK_COLORS = {"Low": "#2E7D32", "Medium": "#FB8C00", "High": "#E53935"}
RISK_ORDER = ["High", "Medium", "Low"]
# Cut-offs used to create Risk_Level in customer_churn_predictions.csv
LOW_CUT, HIGH_CUT = 0.35, 0.65

# Files the dashboard needs (shown in the sidebar status panel)
REQUIRED_FILES = {
    "Daily sales": PROCESSED / "daily_sales.csv",
    "Customer RFM segments": PROCESSED / "customer_rfm.csv",
    "Churn predictions": PROCESSED / "customer_churn_predictions.csv",
    "Churn metrics": REPORTS / "churn_metrics.json",
    "Churn feature importance": REPORTS / "churn_feature_importance.csv",
    "Segment summary": REPORTS / "customer_segment_summary.csv",
    "Forecast model comparison": REPORTS / "model_comparison.csv",
    "Forecast predictions (all models)": REPORTS / "all_model_predictions.csv",
    "Prophet 30-day forecast": REPORTS / "prophet_future_forecast.csv",
    "Inventory recommendations": REPORTS / "inventory_recommendations.csv",
}



# Data loading (cached) - every loader raises if the file is missing

@st.cache_data
def load_daily_sales() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "daily_sales.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_customers() -> pd.DataFrame:
    """RFM segments joined with churn predictions (one row per customer)."""
    rfm = pd.read_csv(PROCESSED / "customer_rfm.csv", parse_dates=["Last_Purchase"])
    pred = pd.read_csv(PROCESSED / "customer_churn_predictions.csv")
    pred["CustomerID"] = pred["Customer ID"].astype(int)
    pred = pred.drop(columns=["Customer ID"])
    return rfm.merge(pred, on="CustomerID", how="left")


@st.cache_data
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data
def load_churn_metrics() -> dict:
    with open(REPORTS / "churn_metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_churn_model():
    """Load models/churn_model.pkl. Returns (bundle, error_message)."""
    path = MODELS / "churn_model.pkl"
    if not path.exists():
        return None, "models/churn_model.pkl was not found."
    try:
        import joblib  # installed together with scikit-learn
    except ImportError:
        return None, "scikit-learn / joblib is not installed (see requirements.txt)."
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = joblib.load(path)
        note = None
        for w in caught:
            if "InconsistentVersion" in type(w.message).__name__:
                note = ("The model was saved with a different scikit-learn version "
                        "than the one installed here, so live scores may differ slightly.")
                break
        model, features = obj["model"], list(obj["features"])
        if not hasattr(model, "predict_proba"):
            raise ValueError("saved object has no predict_proba")
        return {"model": model, "features": features, "note": note}, None
    except Exception as exc:  # corrupted file, version problems, wrong format...
        return None, f"Could not load the churn model ({type(exc).__name__}: {exc})."


def try_load(fn, *args):
    """Run a loader; return (data, error_message)."""
    try:
        return fn(*args), None
    except FileNotFoundError as exc:
        return None, f"Missing file: {Path(str(exc.filename)).name}"
    except Exception as exc:
        return None, f"Could not read data ({type(exc).__name__}: {exc})"


def need(fn, *args):
    """Load data for a page; show a friendly error and stop that page if it fails."""
    data, err = try_load(fn, *args)
    if err:
        st.error(err)
        st.info("Ask the teammate who owns this output to regenerate it, or run the "
                "matching notebook / script in src/, then refresh this page.")
        st.stop()
    return data

# Helpers
def money(x: float) -> str:
    return f"£{x:,.0f}"


def money_short(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"£{x / 1_000_000:.2f}M"
    if abs(x) >= 10_000:
        return f"£{x / 1_000:.0f}K"
    return f"£{x:,.0f}"


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col.container(border=True):
            st.metric(label, value, delta)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def risk_level(score: float) -> str:
    if score >= HIGH_CUT:
        return "High"
    if score <= LOW_CUT:
        return "Low"
    return "Medium"


def file_status() -> dict:
    return {name: path.exists() for name, path in REQUIRED_FILES.items()}



# Sidebar filters (navigation is added by st.navigation at the bottom)
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")

_daily, _daily_err = try_load(load_daily_sales)
_customers, _cust_err = try_load(load_customers)

st.sidebar.subheader("Filters")
if _daily is not None:
    min_d, max_d = _daily["Date"].min().date(), _daily["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Sales date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:  # user is still choosing the second date
        start_d, end_d = min_d, max_d
else:
    start_d = end_d = None
    st.sidebar.warning(_daily_err)

if _customers is not None:
    all_segments = sorted(_customers["Segment_Name"].unique())
    sel_segments = st.sidebar.multiselect("Customer segments", all_segments, default=all_segments)
    sel_risk = st.sidebar.multiselect("Churn risk level", RISK_ORDER, default=RISK_ORDER)
else:
    all_segments, sel_segments, sel_risk = [], [], []
    st.sidebar.warning(_cust_err)

with st.sidebar.expander("System status"):
    for name, ok in file_status().items():
        st.write(f"{'✅' if ok else '❌'} {name}")
    _bundle, _model_err = load_churn_model()
    st.write(f"{'✅' if _bundle else '⚠️'} Churn model (live scoring)")
    st.write(f"{'✅' if (MODELS / 'lstm_model.pth').exists() else '❌'} LSTM weights file")
    st.caption("LSTM results are shown from saved reports; the weights are not run here.")

st.sidebar.divider()
st.sidebar.caption("Data: UCI Online Retail II (Dec 2009 - Dec 2011)")


def filtered_sales(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[(daily["Date"].dt.date >= start_d) & (daily["Date"].dt.date <= end_d)]


def filtered_customers(customers: pd.DataFrame) -> pd.DataFrame:
    return customers[
        customers["Segment_Name"].isin(sel_segments) & customers["Risk_Level"].isin(sel_risk)
    ]


# PAGE 1 - OVERVIEW

def page_overview():
    header("📈 Sales Overview", "Revenue trends for the selected date range (sidebar filter).")
    daily = need(load_daily_sales)
    sales = filtered_sales(daily)
    if sales.empty:
        st.warning("No sales data in the selected date range.")
        return

    total_rev = sales["Total_Revenue"].sum()
    total_inv = sales["Total_Invoices"].sum()
    aov = total_rev / total_inv if total_inv else 0

    days = (end_d - start_d).days + 1
    prev_start = pd.Timestamp(start_d) - pd.Timedelta(days=days)
    prev = daily[(daily["Date"] >= prev_start) & (daily["Date"] < pd.Timestamp(start_d))]
    delta = None
    if len(prev) > 0 and prev["Total_Revenue"].sum() > 0:
        delta = f"{(total_rev / prev['Total_Revenue'].sum() - 1) * 100:+.1f}% vs previous period"

    kpi_row(
        [
            ("Total revenue", money_short(total_rev), delta),
            ("Invoices", f"{total_inv:,.0f}", None),
            ("Units sold", f"{sales['Total_Quantity'].sum():,.0f}", None),
            ("Avg order value", money(aov), None),
            ("Avg daily revenue", money(sales["Total_Revenue"].mean()), None),
        ]
    )

    trend = sales.copy()
    trend["7-day average"] = trend["Total_Revenue"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["Date"], y=trend["Total_Revenue"], name="Daily revenue",
                             mode="lines", line=dict(color="#90CAF9", width=1)))
    fig.add_trace(go.Scatter(x=trend["Date"], y=trend["7-day average"], name="7-day average",
                             mode="lines", line=dict(color="#1565C0", width=2.5)))
    spikes = trend[trend["Revenue_Spike_Flag"]]
    if not spikes.empty:
        fig.add_trace(go.Scatter(x=spikes["Date"], y=spikes["Total_Revenue"], name="Revenue spike",
                                 mode="markers", marker=dict(color="#E53935", size=11, symbol="star")))
    fig.update_layout(title="Daily revenue", yaxis_title="Revenue (£)")
    st.plotly_chart(style(fig, 400))

    c1, c2 = st.columns(2)
    with c1:
        monthly = (
            sales.assign(Month=sales["Date"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month", as_index=False)["Total_Revenue"].sum()
        )
        fig = px.bar(monthly, x="Month", y="Total_Revenue", title="Monthly revenue",
                     labels={"Total_Revenue": "Revenue (£)"})
        fig.update_traces(marker_color="#1E88E5")
        st.plotly_chart(style(fig))
    with c2:
        dow = (
            sales.assign(Weekday=sales["Date"].dt.day_name())
            .groupby("Weekday", as_index=False)["Total_Revenue"].mean()
        )
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow["Weekday"] = pd.Categorical(dow["Weekday"], categories=order, ordered=True)
        fig = px.bar(dow.sort_values("Weekday"), x="Weekday", y="Total_Revenue",
                     title="Average revenue by weekday", labels={"Total_Revenue": "Avg revenue (£)"})
        fig.update_traces(marker_color="#43A047")
        st.plotly_chart(style(fig))
    st.caption("The data ends on 9 Dec 2011, so the last day may be incomplete.")



# PAGE 2 - CUSTOMER SEGMENTS
def page_segments():
    header("👥 Customer Segments", "RFM segmentation results. Uses the segment and risk filters in the sidebar.")
    customers = need(load_customers)
    summary = need(load_report, "customer_segment_summary.csv")
    cust = filtered_customers(customers)
    if cust.empty:
        st.warning("No customers match the current sidebar filters.")
        return

    total_spend = cust["Monetary"].sum()
    champ_spend = cust.loc[cust["Segment_Name"] == "Champions", "Monetary"].sum()
    kpi_row(
        [
            ("Customers", f"{len(cust):,}", None),
            ("Total historical spend", money_short(total_spend), None),
            ("Avg spend / customer", money(cust["Monetary"].mean()), None),
            ("Avg orders / customer", f"{cust['Frequency'].mean():.1f}", None),
            ("Champions' share of spend", f"{champ_spend / total_spend * 100:.0f}%" if total_spend else "-", None),
        ]
    )

    by_seg = (
        cust.groupby("Segment_Name")
        .agg(Customers=("CustomerID", "count"), Spend=("Monetary", "sum"),
             Avg_Recency=("Recency", "mean"), Avg_Frequency=("Frequency", "mean"),
             Avg_Monetary=("Monetary", "mean"))
        .reset_index().sort_values("Customers", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_seg, x="Segment_Name", y="Customers", color="Segment_Name",
                     color_discrete_map=SEGMENT_COLORS, title="Customers per segment")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style(fig))
    with c2:
        fig = px.pie(by_seg, names="Segment_Name", values="Spend", hole=0.5,
                     color="Segment_Name", color_discrete_map=SEGMENT_COLORS,
                     title="Share of historical spend")
        st.plotly_chart(style(fig))

    st.subheader("Frequency vs spend")
    sample = cust.sample(min(len(cust), 2500), random_state=42)
    fig = px.scatter(sample, x="Frequency", y="Monetary", color="Segment_Name",
                     color_discrete_map=SEGMENT_COLORS, log_x=True, log_y=True, opacity=0.6,
                     hover_data=["CustomerID", "Recency"],
                     labels={"Monetary": "Total spend (£)", "Frequency": "Orders"})
    st.plotly_chart(style(fig, 450))
    st.caption("Log scale on both axes. A random sample of up to 2,500 customers is shown.")

    st.subheader("Segment profile")
    st.dataframe(by_seg.rename(columns={"Segment_Name": "Segment"}).round(1), hide_index=True)
    st.download_button("Download segment profile (CSV)", by_seg.to_csv(index=False),
                       file_name="segment_profile.csv", mime="text/csv")

    with st.expander("Segment summary from the segmentation notebook"):
        st.dataframe(summary, hide_index=True)



# PAGE 3 - CHURN PREDICTION
def page_churn():
    header("⚠️ Churn Prediction", "Risk scores and evaluation metrics from the RandomForest churn model.")
    customers = need(load_customers)
    m = need(load_churn_metrics)
    feat_imp = need(load_report, "churn_feature_importance.csv")
    cust = filtered_customers(customers)

    kpi_row(
        [
            ("Accuracy", f"{m['accuracy']:.1%}", None),
            ("Precision", f"{m['precision']:.1%}", None),
            ("Recall", f"{m['recall']:.1%}", None),
            ("ROC-AUC", f"{m['roc_auc']:.3f}", None),
        ]
    )
    st.warning(
        "**Current model limitations**  \n"
        f"- {m['target_definition']}  \n"
        f"- {m['note']}  \n"
        f"- The model uses only {len(m['features_used'])} inputs ({', '.join(m['features_used'])}). "
        "Metrics around 69% accuracy and 0.74 ROC-AUC are moderate, not production-grade.  \n"
        "- Saved risk scores cover all customers, including those the model was trained on."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(m["confusion_matrix"], text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted: Stay", "Predicted: Churn"], y=["Actual: Stay", "Actual: Churn"],
                        title="Confusion matrix (test set)")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style(fig))
    with c2:
        fig = px.bar(feat_imp.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Feature importance")
        fig.update_traces(marker_color="#8E24AA")
        st.plotly_chart(style(fig))

    st.divider()
    st.subheader("Customer risk explorer")
    st.caption("Uses the segment and risk filters in the sidebar.")
    if cust.empty:
        st.warning("No customers match the current sidebar filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            counts = cust["Risk_Level"].value_counts().reindex(RISK_ORDER).dropna().reset_index()
            counts.columns = ["Risk_Level", "Customers"]
            fig = px.bar(counts, x="Risk_Level", y="Customers", color="Risk_Level",
                         color_discrete_map=RISK_COLORS, title="Customers by risk level")
            fig.update_layout(showlegend=False, xaxis_title=None)
            st.plotly_chart(style(fig))
        with c2:
            fig = px.histogram(cust, x="Churn_Risk_Score", nbins=30, title="Risk score distribution")
            fig.update_traces(marker_color="#E53935")
            st.plotly_chart(style(fig))

        st.markdown("**Priority list: highest-spending customers at high risk**")
        top_n = st.slider("Number of customers", 10, 200, 25, step=5)
        priority = (
            cust[cust["Risk_Level"] == "High"].sort_values("Monetary", ascending=False).head(top_n)
            [["CustomerID", "Segment_Name", "Monetary", "Frequency", "Recency", "Churn_Risk_Score", "Risk_Level"]]
        )
        st.dataframe(
            priority, hide_index=True,
            column_config={
                "Monetary": st.column_config.NumberColumn("Total spend", format="£%.0f"),
                "Churn_Risk_Score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.download_button("Download priority list (CSV)", priority.to_csv(index=False),
                           file_name="high_risk_customers.csv", mime="text/csv")

    st.divider()
    st.subheader("Live scoring with the saved model")
    bundle, err = load_churn_model()
    if bundle is None:
        st.error(err)
        st.info("Live scoring is unavailable. Everything else on this page uses the saved predictions.")
    else:
        if bundle["note"]:
            st.caption(f"⚠️ {bundle['note']}")
        a, b, c = st.columns(3)
        orders = a.number_input("Total orders", min_value=1, value=5, step=1)
        spend = b.number_input("Total spending (£)", min_value=0.0, value=2000.0, step=100.0)
        qty = c.number_input("Total quantity (units)", min_value=0, value=1200, step=50)
        try:
            row = pd.DataFrame([{"Total_Orders": orders, "Total_Spending": spend,
                                 "Total_Quantity": qty}])[bundle["features"]]
            classes = list(bundle["model"].classes_)
            score = float(bundle["model"].predict_proba(row)[0][classes.index(1)])
            level = risk_level(score)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score * 100,
                number={"suffix": "%"}, title={"text": f"Churn risk: {level}"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[level]},
                       "steps": [{"range": [0, LOW_CUT * 100], "color": "#E8F5E9"},
                                 {"range": [LOW_CUT * 100, HIGH_CUT * 100], "color": "#FFF3E0"},
                                 {"range": [HIGH_CUT * 100, 100], "color": "#FFEBEE"}]},
            ))
            st.plotly_chart(style(fig, 280))
        except Exception as exc:
            st.error(f"Scoring failed ({type(exc).__name__}: {exc}).")

    st.divider()
    st.subheader("Look up a customer")
    cid = st.number_input("Customer ID", min_value=0, value=int(customers["CustomerID"].iloc[0]), step=1)
    row = customers[customers["CustomerID"] == cid]
    if row.empty:
        st.caption("No customer found with this ID.")
    else:
        r = row.iloc[0]
        kpi_row(
            [
                ("Segment", r["Segment_Name"], None),
                ("Total spend", money(r["Monetary"]), None),
                ("Orders", f"{r['Frequency']:.0f}", None),
                ("Days since last order", f"{r['Recency']:.0f}", None),
                ("Churn risk", f"{r['Churn_Risk_Score']:.0%} ({r['Risk_Level']})", None),
            ]
        )

    st.divider()
    st.subheader("What-if: retention campaign")
    high_spend = customers.loc[customers["Risk_Level"] == "High", "Monetary"].sum()
    w1, w2 = st.columns(2)
    retain_pct = w1.slider("Share of high-risk customers retained (%)", 0, 100, 20)
    w2.metric("Historical spend of all high-risk customers", money_short(high_spend))
    st.metric("Historical spend protected", money_short(high_spend * retain_pct / 100),
              help="Based on past total spend. This is not a revenue forecast.")

# PAGE 4 - DEMAND FORECAST
def page_forecast():
    header("🔮 Demand Forecast", "Prophet and LSTM revenue forecasts compared with a naive baseline.")
    model_cmp = need(load_report, "model_comparison.csv")
    all_pred = need(load_report, "all_model_predictions.csv")
    future = need(load_report, "prophet_future_forecast.csv")
    all_pred["Date"] = pd.to_datetime(all_pred["Date"])
    future["ds"] = pd.to_datetime(future["ds"])

    mape = dict(zip(model_cmp["Model"], model_cmp["MAPE_percent"]))
    naive, prophet, lstm = mape.get("Naive Baseline"), mape.get("Prophet"), mape.get("LSTM")

    st.subheader("Model comparison (30-day hold-out)")
    best = model_cmp.sort_values("MAPE_percent").iloc[0]
    kpi_row(
        [
            ("Lowest-error model", best["Model"], None),
            ("MAE", money(best["MAE"]), None),
            ("RMSE", money(best["RMSE"]), None),
            ("MAPE", f"{best['MAPE_percent']:.1f}%", None),
        ]
    )

    notes = []
    if prophet is not None and naive is not None:
        notes.append(f"Prophet MAPE is {prophet:.1f}% against {naive:.1f}% for the naive baseline. "
                     "It is better, but a 30% average error is still large for daily revenue.")
    if lstm is not None and naive is not None and lstm > naive:
        notes.append(f"The LSTM (MAPE {lstm:.1f}%) does not beat the naive baseline ({naive:.1f}%).")
    alt, _ = try_load(load_report, "lstm_metrics.csv")
    if alt is not None and lstm is not None and abs(float(alt["MAPE_percent"].iloc[0]) - lstm) > 0.5:
        notes.append(f"LSTM results are inconsistent between report files: lstm_metrics.csv says "
                     f"MAPE {float(alt['MAPE_percent'].iloc[0]):.1f}%, model_comparison.csv says {lstm:.1f}%. "
                     "This needs to be confirmed with the forecasting owner.")
    notes.append("Evaluation uses a single 30-day hold-out window, so results may not hold in other periods.")
    st.warning("**Current model limitations**\n" + "\n".join(f"- {n}" for n in notes))

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(model_cmp.round(1), hide_index=True,
                     column_config={"MAPE_percent": st.column_config.NumberColumn("MAPE %")})
    with c2:
        fig = px.bar(model_cmp, x="Model", y="MAPE_percent", color="Model",
                     title="Forecast error (MAPE %, lower is better)")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style(fig, 300))

    st.subheader("Actual vs forecast (hold-out period)")
    cols = [c for c in all_pred.columns if c.endswith("_Forecast")]
    chosen = st.multiselect("Models to show", cols, default=cols,
                            format_func=lambda c: c.replace("_Forecast", ""))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_pred["Date"], y=all_pred["Actual_Revenue"], name="Actual",
                             mode="lines+markers", line=dict(color="black", width=3)))
    for col in chosen:
        fig.add_trace(go.Scatter(x=all_pred["Date"], y=all_pred[col],
                                 name=col.replace("_Forecast", ""), mode="lines"))
    fig.update_layout(yaxis_title="Revenue (£)")
    st.plotly_chart(style(fig, 420))

    st.subheader("Next 30 days (Prophet)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(30,136,229,0.2)", name="Uncertainty range"))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines+markers",
                             name="Forecast", line=dict(color="#1565C0", width=2.5)))
    fig.update_layout(yaxis_title="Revenue (£)")
    st.plotly_chart(style(fig, 400))
    st.metric("Forecast revenue, next 30 days", money(future["yhat"].clip(lower=0).sum()))
    st.caption("Only Prophet produces the forward forecast. The wide range shows real uncertainty.")
    st.download_button("Download 30-day forecast (CSV)", future.to_csv(index=False),
                       file_name="forecast_30_days.csv", mime="text/csv")


# PAGE 5 - INVENTORY

def page_inventory():
    header("📦 Inventory Recommendations", "Reorder planning based on the 30-day Prophet forecast.")
    inv = need(load_report, "inventory_recommendations.csv")
    future = need(load_report, "prophet_future_forecast.csv")
    inv["Forecast_Date"] = pd.to_datetime(inv["Forecast_Date"])
    base = inv.iloc[0]

    st.warning(
        "**Current limitations**  \n"
        "- Recommendations are at total-business level, not per product (SKU).  \n"
        "- Lead time and the safety-stock rule are assumptions, not supplier data.  \n"
        f"- Every row currently gives the same advice: \"{base['Recommendation']}\"."
    )

    kpi_row(
        [
            ("Avg daily units", f"{base['Estimated_Avg_Daily_Units']:,.0f}", None),
            ("Lead time (days)", f"{base['Assumed_Lead_Time_Days']:.0f}", None),
            ("Safety stock (units)", f"{base['Safety_Stock_Units']:,.0f}", None),
            ("Reorder point (units)", f"{base['Reorder_Point_Units']:,.0f}", None),
        ]
    )

    d_min, d_max = inv["Forecast_Date"].min().date(), inv["Forecast_Date"].max().date()
    rng = st.date_input("Forecast date range", value=(d_min, d_max), min_value=d_min, max_value=d_max)
    if isinstance(rng, tuple) and len(rng) == 2:
        shown = inv[(inv["Forecast_Date"].dt.date >= rng[0]) & (inv["Forecast_Date"].dt.date <= rng[1])]
    else:
        shown = inv

    fig = px.bar(shown, x="Forecast_Date", y="Forecast_Revenue", title="Forecast revenue by day",
                 labels={"Forecast_Revenue": "Revenue (£)"})
    fig.update_traces(marker_color="#1E88E5")
    st.plotly_chart(style(fig, 320))

    st.dataframe(
        shown, hide_index=True,
        column_config={
            "Forecast_Date": st.column_config.DateColumn("Date"),
            "Forecast_Revenue": st.column_config.NumberColumn("Forecast revenue", format="£%.0f"),
        },
    )
    st.download_button("Download inventory recommendations (CSV)", shown.to_csv(index=False),
                       file_name="inventory_recommendations.csv", mime="text/csv")

    st.divider()
    st.subheader("What-if: demand and supply scenario")
    s1, s2, s3 = st.columns(3)
    demand_change = s1.slider("Demand change (%)", -30, 50, 0)
    lead_time = s2.slider("Supplier lead time (days)", 1, 30, int(base["Assumed_Lead_Time_Days"]))
    safety_days = s3.slider("Safety stock (days of demand)", 0, 10, 3)

    base_units = base["Estimated_Avg_Daily_Units"]
    new_units = base_units * (1 + demand_change / 100)
    new_safety = new_units * safety_days
    new_reorder = new_units * lead_time + new_safety
    base_rev = future["yhat"].clip(lower=0).sum()
    new_rev = base_rev * (1 + demand_change / 100)

    kpi_row(
        [
            ("Daily units", f"{new_units:,.0f}", f"{new_units - base_units:+,.0f}"),
            ("Safety stock", f"{new_safety:,.0f}", f"{new_safety - base['Safety_Stock_Units']:+,.0f}"),
            ("Reorder point", f"{new_reorder:,.0f}", f"{new_reorder - base['Reorder_Point_Units']:+,.0f}"),
            ("30-day revenue", money_short(new_rev), f"{new_rev - base_rev:+,.0f}"),
        ]
    )
    st.caption("Reorder point = daily units x lead time + safety stock. Check product-level stock before ordering.")



# PAGE 6 - MODEL STATUS & LIMITATIONS

def build_status_table():
    m, _ = try_load(load_churn_metrics)
    cmp_, _ = try_load(load_report, "model_comparison.csv")
    seg, _ = try_load(load_report, "customer_segment_summary.csv")
    customers, _ = try_load(load_customers)
    mape = dict(zip(cmp_["Model"], cmp_["MAPE_percent"])) if cmp_ is not None else {}
    bundle, _ = load_churn_model()

    rows = [
        {"Component": "Customer segmentation (RFM)", "Status": "Integrated",
         "Current result": (f"{customers['Segment_Name'].nunique()} segments, {len(customers):,} customers"
                            if customers is not None else "n/a"),
         "Limitation": "Rule-based snapshot of historical data; segments are not predictions."},
        {"Component": "Churn model (RandomForest)",
         "Status": "Integrated" if bundle else "Saved results only",
         "Current result": (f"Accuracy {m['accuracy']:.1%}, ROC-AUC {m['roc_auc']:.3f}" if m else "n/a"),
         "Limitation": "Proxy churn label, 3 input features, random (not time-based) split."},
        {"Component": "Prophet forecast", "Status": "Integrated",
         "Current result": (f"MAPE {mape['Prophet']:.1f}% (naive {mape['Naive Baseline']:.1f}%)"
                            if "Prophet" in mape and "Naive Baseline" in mape else "n/a"),
         "Limitation": "30-day hold-out only; error is still high on volatile daily revenue."},
        {"Component": "LSTM forecast", "Status": "Needs team confirmation",
         "Current result": f"MAPE {mape['LSTM']:.1f}%" if "LSTM" in mape else "n/a",
         "Limitation": "Saved results only; does not beat the naive baseline; report files disagree."},
        {"Component": "Inventory recommendations", "Status": "Integrated",
         "Current result": "Reorder point from the 30-day forecast",
         "Limitation": "Business-level, not per product; assumed lead time and safety stock."},
    ]
    _ = seg
    return pd.DataFrame(rows)


def page_status():
    header("🧪 Model Status & Limitations",
           "What is integrated, what the results really are, and what is still open.")
    st.info("This dashboard reports current results only. No accuracy targets are claimed as "
            "achieved, and every figure comes from the saved project outputs.")

    st.subheader("Component status")
    st.dataframe(build_status_table(), hide_index=True)

    st.subheader("Details")
    with st.expander("Churn prediction", expanded=True):
        st.markdown(
            "- The churn label is supplied with the data. It may be a rule such as no purchase in the last "
            "90 days, which would make it a proxy rather than observed future churn.\n"
            "- Recency and last purchase date are excluded to avoid target leakage, so the model sees only "
            "orders, spending and quantity.\n"
            "- Metrics come from one random split of 5,878 customers. Validate on a time-based split before "
            "using scores for decisions.\n"
            "- About half of all customers carry the churn label, so the baseline is hard to judge."
        )
    with st.expander("Demand forecasting", expanded=True):
        st.markdown(
            "- Prophet is the best of the three models tested, but still has a high average error.\n"
            "- The LSTM is worse than the naive baseline in `model_comparison.csv`.\n"
            "- `lstm_metrics.csv` and `lstm_forecast_results.csv` do not match `model_comparison.csv` and "
            "`all_model_predictions.csv`. The dashboard uses the latter two.\n"
            "- Daily revenue has weekly patterns and near-zero days, which makes percentage error large."
        )
    with st.expander("Inventory", expanded=True):
        st.markdown(
            "- Output is total units per day, not per product.\n"
            "- All 30 rows have the same units, lead time, safety stock and advice.\n"
            "- Real stock levels and supplier lead times are not part of the data."
        )
    with st.expander("Data"):
        st.markdown(
            "- UCI Online Retail II: one UK online retailer, Dec 2009 to 9 Dec 2011. This is historical "
            "data, not a live feed.\n"
            "- Customer-level analysis covers only customers with an ID."
        )

    st.subheader("Open items for the team")
    st.markdown(
        "1. Confirm the correct LSTM results (two sets of numbers exist).\n"
        "2. Confirm how the churn label was created and add a time-based validation.\n"
        "3. Share product-level data if SKU-level inventory advice is wanted.\n"
        "4. Keep the scikit-learn version in `requirements.txt` the same as the one used to train the model."
    )



# PAGE 7 - EXPORT

def page_export():
    header("📥 Reports & Export", "Download the filtered data and a short summary report.")
    daily = need(load_daily_sales)
    customers = need(load_customers)
    future = need(load_report, "prophet_future_forecast.csv")
    cmp_ = need(load_report, "model_comparison.csv")
    m = need(load_churn_metrics)
    sales, cust = filtered_sales(daily), filtered_customers(customers)
    best = cmp_.sort_values("MAPE_percent").iloc[0]

    summary = "\n".join(
        [
            "# RetailPulse Summary Report",
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            "",
            "## Sales",
            f"- Period: {start_d} to {end_d}",
            f"- Revenue: {money(sales['Total_Revenue'].sum())}",
            f"- Invoices: {sales['Total_Invoices'].sum():,.0f}",
            "",
            "## Customers",
            f"- Total customers: {len(customers):,}",
            f"- High churn risk: {(customers['Risk_Level'] == 'High').sum():,}",
            "",
            "## Forecast",
            f"- Lowest-error model: {best['Model']} (MAPE {best['MAPE_percent']:.1f}%)",
            f"- Next 30 days forecast revenue: {money(future['yhat'].clip(lower=0).sum())}",
            "",
            "## Current limitations",
            f"- Churn: {m['accuracy']:.1%} accuracy, ROC-AUC {m['roc_auc']:.3f}. {m['note']}",
            "- Forecast: single 30-day hold-out; LSTM does not beat the naive baseline.",
            "- Inventory: business-level, not per product; assumptions for lead time and safety stock.",
        ]
    )
    st.download_button("📄 Summary report (Markdown)", summary,
                       file_name="retailpulse_summary.md", mime="text/markdown")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Daily sales (filtered)", sales.to_csv(index=False),
                       file_name="daily_sales_filtered.csv", mime="text/csv")
    c2.download_button("Customers + churn risk (filtered)", cust.to_csv(index=False),
                       file_name="customers_filtered.csv", mime="text/csv")
    c3.download_button("30-day forecast", future.to_csv(index=False),
                       file_name="forecast_30_days.csv", mime="text/csv")
    with st.expander("Preview summary report"):
        st.markdown(summary)



# Sidebar navigation

navigation = st.navigation(
    [
        st.Page(page_overview, title="Overview", icon="📈", url_path="overview", default=True),
        st.Page(page_segments, title="Customer Segments", icon="👥", url_path="segments"),
        st.Page(page_churn, title="Churn Prediction", icon="⚠️", url_path="churn"),
        st.Page(page_forecast, title="Demand Forecast", icon="🔮", url_path="forecast"),
        st.Page(page_inventory, title="Inventory", icon="📦", url_path="inventory"),
        st.Page(page_status, title="Status & Limitations", icon="🧪", url_path="status"),
        st.Page(page_export, title="Reports & Export", icon="📥", url_path="export"),
    ]
)
navigation.run()