"""
Streamlit dashboard for New York State industry compensation forecasts.

Expected repository layout:
    project_root/
    ├── app.py                  # this file
    └── data.csv                # BEA quarterly compensation file

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NYS Industry Compensation Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

PALETTE = [
    "#300049",  # purple
    "#00838F",  # teal
    "#D47A1F",  # orange
    "#C2185B",  # magenta
]

TEST_LENGTH = 20
MAX_HORIZON = 24
RF_LAGS = (1, 2, 3, 4, 8, 12)

SERIES_LABELS = {
    "finance": "Finance",
    "health": "Health care",
    "prof_svcs": "Professional/Technical services",
    "tech": "Technology",
}

DISPLAY_TO_KEY = {v: k for k, v in SERIES_LABELS.items()}

CHOSEN_ARIMA_ORDERS = {
    "finance": (0, 1, 2),
    "health": (3, 1, 3),
    "prof_svcs": (1, 1, 1),
    "tech": (0, 2, 2),
}

INDUSTRY_MAP = {
    "Wholesale trade": "Wholesale trade",
    "Retail trade": "Retail trade",
    "Transportation and warehousing": "Transport",
    "Information": "Technology",
    "Finance and insurance": "Finance",
    "Real estate and rental and leasing": "Real estate",
    "Professional, scientific, and technical services": "Professional/Technical services",
    "Management of companies and enterprises": "Management",
    "Administrative and support and waste management and remediation services": "Admin/support services",
    "Educational services": "Education",
    "Health care and social assistance": "Health care",
    "Arts, entertainment, and recreation": "Entertainment",
    "Accommodation and food services": "Hospitality",
    "Other services (except government and government enterprises)": "Other private services",
}

TARGET_INDUSTRIES = [
    "Finance",
    "Health care",
    "Professional/Technical services",
    "Technology",
]


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def dollars_millions(x: float, pos: int | None = None) -> str:
    return f"${x / 1e6:,.0f}M"


def style_metric_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return df.style.format(
        {
            "MAE": "{:,.0f}",
            "RMSE": "{:,.0f}",
            "MAPE": "{:.2%}",
            "R2": "{:.3f}",
            "Ljung-Box p-value": "{:.4f}",
        }
    )


# -----------------------------------------------------------------------------
# Data loading and transformations
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data(data_path: str | Path = "data.csv") -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Place data.csv in the same folder as app.py."
        )

    raw = pd.read_csv(data_path, skiprows=3)

    if "Description" not in raw.columns:
        raise ValueError("Expected a BEA-style file with a 'Description' column.")

    drop_cols = [c for c in ["GeoFIPS", "GeoName"] if c in raw.columns]

    df = (
        raw.dropna(subset=["Description"])
        .drop(columns=drop_cols)
        .set_index("LineCode")
        .melt(
            id_vars=["Description"],
            var_name="period",
            value_name="compensation",
            ignore_index=False,
        )
    )

    df.index.name = "code"
    df = df.rename(columns={"Description": "industry"})

    df["industry_raw"] = df["industry"].astype(str).str.strip()
    df["industry"] = df["industry_raw"].map(INDUSTRY_MAP)

    # Convert BEA period strings such as 1998:Q1 to quarter-start timestamps.
    period_str = df["period"].astype(str)
    df["period"] = (
        period_str.str.replace(r"(\d{4}):Q1", r"\1-01-01", regex=True)
        .str.replace(r"(\d{4}):Q2", r"\1-04-01", regex=True)
        .str.replace(r"(\d{4}):Q3", r"\1-07-01", regex=True)
        .str.replace(r"(\d{4}):Q4", r"\1-10-01", regex=True)
    )
    df["period"] = pd.to_datetime(df["period"], errors="coerce")

    df["compensation"] = (
        df["compensation"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace({"(NA)": np.nan, "NA": np.nan, "nan": np.nan})
    )
    df["compensation"] = pd.to_numeric(df["compensation"], errors="coerce")

    df = df.dropna(subset=["industry", "period", "compensation"]).copy()
    df = df.loc[df["period"] != pd.Timestamp("2025-10-01")].copy()
    df["year"] = df["period"].dt.year

    df["industry_group"] = np.where(
        df["industry"].isin(TARGET_INDUSTRIES), df["industry"], "Other"
    )

    return df.sort_values(["industry", "period"]).reset_index()


@st.cache_data(show_spinner=False)
def build_target_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    series_dict: dict[str, pd.Series] = {}

    for key, label in SERIES_LABELS.items():
        s = (
            df.loc[df["industry"] == label]
            .groupby("period")["compensation"]
            .sum()
            .sort_index()
            .asfreq("QS")
        )
        s.name = key
        series_dict[key] = s

    return series_dict


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------


def train_test_split(series: pd.Series, test_n: int = TEST_LENGTH) -> tuple[pd.Series, pd.Series]:
    return series.iloc[:-test_n], series.iloc[-test_n:]


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_pred = pd.Series(y_pred, index=y_true.index)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def fit_ets_forecast(train_series: pd.Series, horizon: int) -> tuple[pd.Series, object]:
    model = ExponentialSmoothing(
        train_series,
        trend="mul",
        damped_trend=False,
        seasonal=None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(horizon)
    return forecast, fit


def build_rf_features(series: pd.Series, lags: tuple[int, ...] = RF_LAGS) -> pd.DataFrame:
    """Build supervised-learning features on first differences of log levels."""
    if (series <= 0).any():
        raise ValueError("Log-difference Random Forest requires strictly positive series values.")

    log_series = np.log(series)
    log_diff_series = log_series.diff()

    out = pd.DataFrame({"y": log_diff_series}, index=series.index)

    for lag in lags:
        out[f"lag_{lag}"] = log_diff_series.shift(lag)

    out["quarter"] = out.index.quarter
    out["quarter_sin"] = np.sin(2 * np.pi * out["quarter"] / 4)
    out["quarter_cos"] = np.cos(2 * np.pi * out["quarter"] / 4)

    prior = log_diff_series.shift(1)
    out["4q_mean_diff"] = prior.rolling(4).mean()
    out["4q_std_diff"] = prior.rolling(4).std()
    out["yoy_diff"] = log_diff_series.shift(4)

    return out.dropna()


def make_next_rf_features(
    history: pd.Series,
    next_date: pd.Timestamp,
    lags: tuple[int, ...] = RF_LAGS,
) -> pd.DataFrame:
    log_history = np.log(history)
    log_diff_history = log_history.diff()

    row: dict[str, float] = {}

    for lag in lags:
        row[f"lag_{lag}"] = log_diff_history.iloc[-lag]

    quarter = next_date.quarter
    row["quarter"] = quarter
    row["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    row["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    prior = log_diff_history.dropna()
    row["4q_mean_diff"] = prior.iloc[-4:].mean()
    row["4q_std_diff"] = prior.iloc[-4:].std()
    row["yoy_diff"] = log_diff_history.iloc[-4]

    return pd.DataFrame(row, index=[next_date])


def recursive_rf_forecast(
    model: RandomForestRegressor,
    train_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
    lags: tuple[int, ...] = RF_LAGS,
) -> pd.Series:
    history = train_series.copy()
    level_preds: list[float] = []

    for date in forecast_index:
        X_next = make_next_rf_features(history, date, lags=lags)
        log_diff_pred = model.predict(X_next)[0]
        next_level = history.iloc[-1] * np.exp(log_diff_pred)

        level_preds.append(next_level)
        history.loc[date] = next_level

    return pd.Series(level_preds, index=forecast_index, name="Random Forest")


def fit_rf_model(train_series: pd.Series) -> RandomForestRegressor:
    feature_matrix = build_rf_features(train_series)
    X_train = feature_matrix.drop(columns="y")
    y_train = feature_matrix["y"]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def residual_interval(
    forecast: pd.Series,
    residuals: pd.Series,
    z_value: float = 1.96,
) -> tuple[pd.Series, pd.Series]:
    sigma = residuals.std(ddof=1)
    lower = forecast - z_value * sigma
    upper = forecast + z_value * sigma
    return lower, upper


# -----------------------------------------------------------------------------
# Model calculations
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=True)
def run_models(data_path: str = "data.csv") -> dict[str, object]:
    df = load_data(data_path)
    series_dict = build_target_series(df)

    ets_models: dict[str, object] = {}
    arima_models: dict[str, object] = {}
    rf_models: dict[str, RandomForestRegressor] = {}

    holdout_forecasts: dict[str, pd.DataFrame] = {}
    future_forecasts: dict[str, pd.DataFrame] = {}
    future_intervals: dict[str, dict[str, pd.DataFrame]] = {}
    residual_data: dict[str, pd.DataFrame] = {}
    error_tables: dict[str, pd.DataFrame] = {}
    ljung_box_tables: dict[str, pd.DataFrame] = {}

    for key, series in series_dict.items():
        train, test = train_test_split(series, TEST_LENGTH)
        forecast_index = test.index

        # ETS holdout model
        ets_fc, ets_fit = fit_ets_forecast(train, TEST_LENGTH)
        ets_fc.index = forecast_index
        ets_models[key] = ets_fit

        # ARIMA holdout model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            arima_fit = ARIMA(
                train,
                order=CHOSEN_ARIMA_ORDERS[key],
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(method_kwargs={"maxiter": 1000})
        arima_models[key] = arima_fit
        arima_pred = arima_fit.get_forecast(steps=TEST_LENGTH)
        arima_fc = arima_pred.predicted_mean
        arima_fc.index = forecast_index

        # Random Forest holdout model
        rf_fit = fit_rf_model(train)
        rf_models[key] = rf_fit
        rf_fc = recursive_rf_forecast(rf_fit, train, forecast_index)

        holdout_forecasts[key] = pd.DataFrame(
            {
                "Actual": test,
                "ETS": ets_fc,
                "ARIMA": arima_fc,
                "Random Forest": rf_fc,
            },
            index=forecast_index,
        )

        residual_data[key] = pd.DataFrame(
            {
                "ETS": test - ets_fc,
                "ARIMA": test - arima_fc,
                "Random Forest": test - rf_fc,
            },
            index=forecast_index,
        )

        error_tables[key] = pd.DataFrame(
            [
                {"Model": "ETS", **evaluate(test, ets_fc)},
                {"Model": "ARIMA", **evaluate(test, arima_fc)},
                {"Model": "Random Forest", **evaluate(test, rf_fc)},
            ]
        ).set_index("Model").sort_values("MAPE")

        lb_rows = []
        for model_name in ["ETS", "ARIMA", "Random Forest"]:
            resid = residual_data[key][model_name].dropna()
            if len(resid) > 10:
                lb = acorr_ljungbox(resid, lags=[10], return_df=True)
                lb_pvalue = float(lb["lb_pvalue"].iloc[0])
            else:
                lb_pvalue = np.nan
            lb_rows.append({"Model": model_name, "Ljung-Box p-value": lb_pvalue})
        ljung_box_tables[key] = pd.DataFrame(lb_rows).set_index("Model")

        # Future forecasts are trained on the full available series.
        future_index = pd.date_range(
            series.index.max() + pd.offsets.QuarterBegin(1),
            periods=MAX_HORIZON,
            freq="QS",
        )

        ets_future, _ = fit_ets_forecast(series, MAX_HORIZON)
        ets_future.index = future_index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            arima_future_fit = ARIMA(
                series,
                order=CHOSEN_ARIMA_ORDERS[key],
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(method_kwargs={"maxiter": 1000})
        arima_future_pred = arima_future_fit.get_forecast(steps=MAX_HORIZON)
        arima_future = arima_future_pred.predicted_mean
        arima_future.index = future_index

        rf_full_fit = fit_rf_model(series)
        rf_future = recursive_rf_forecast(rf_full_fit, series, future_index)

        future_forecasts[key] = pd.DataFrame(
            {
                "ETS": ets_future,
                "ARIMA": arima_future,
                "Random Forest": rf_future,
            },
            index=future_index,
        )

        future_intervals[key] = {}
        for model_name in ["ETS", "ARIMA", "Random Forest"]:
            lower, upper = residual_interval(
                future_forecasts[key][model_name],
                residual_data[key][model_name].dropna(),
            )
            future_intervals[key][model_name] = pd.DataFrame(
                {"lower": lower, "upper": upper}, index=future_index
            )

    return {
        "df": df,
        "series_dict": series_dict,
        "ets_models": ets_models,
        "arima_models": arima_models,
        "rf_models": rf_models,
        "holdout_forecasts": holdout_forecasts,
        "future_forecasts": future_forecasts,
        "future_intervals": future_intervals,
        "residual_data": residual_data,
        "error_tables": error_tables,
        "ljung_box_tables": ljung_box_tables,
    }


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def plot_actuals_and_forecast(
    historical: pd.Series,
    future_fc: pd.DataFrame,
    intervals: dict[str, pd.DataFrame],
    models: list[str],
    horizon: int,
    start_date: pd.Timestamp,
    title: str,
) -> plt.Figure:
    hist = historical.loc[historical.index >= start_date]
    fc = future_fc.iloc[:horizon]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist.index, hist.values, label="Actual", color=PALETTE[0], linewidth=2.8)

    model_colors = {
        "ETS": PALETTE[1],
        "ARIMA": PALETTE[2],
        "Random Forest": PALETTE[3],
    }

    for model_name in models:
        ax.plot(
            fc.index,
            fc[model_name],
            label=f"{model_name} forecast",
            linestyle="--",
            marker="o",
            color=model_colors.get(model_name),
        )
        interval = intervals[model_name].iloc[:horizon]
        ax.fill_between(
            interval.index,
            interval["lower"].values,
            interval["upper"].values,
            alpha=0.12,
            color=model_colors.get(model_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Compensation")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollars_millions))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_holdout_fit(holdout: pd.DataFrame, models: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(holdout.index, holdout["Actual"], label="Actual", color=PALETTE[0], linewidth=2.8)

    model_colors = {
        "ETS": PALETTE[1],
        "ARIMA": PALETTE[2],
        "Random Forest": PALETTE[3],
    }
    for model_name in models:
        ax.plot(
            holdout.index,
            holdout[model_name],
            label=model_name,
            linestyle="--",
            marker="o",
            color=model_colors.get(model_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Compensation")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollars_millions))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(residuals: pd.Series, title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    residuals.plot(ax=axes[0, 0], color="tab:gray", linewidth=1.5)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Residuals over time")

    axes[0, 1].hist(residuals.dropna(), bins=16, edgecolor="black", alpha=0.75)
    axes[0, 1].set_title("Residual distribution")

    max_lags = max(1, min(12, len(residuals.dropna()) // 2 - 1))
    plot_acf(residuals.dropna(), lags=max_lags, ax=axes[1, 0])
    axes[1, 0].set_title("Residual ACF")

    plot_pacf(residuals.dropna(), lags=max_lags, ax=axes[1, 1], method="ywm")
    axes[1, 1].set_title("Residual PACF")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_industry_overview(df: pd.DataFrame) -> plt.Figure:
    grouped = (
        df.groupby(["period", "industry_group"], as_index=False)["compensation"]
        .sum()
        .sort_values("period")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for group in TARGET_INDUSTRIES + ["Other"]:
        temp = grouped.loc[grouped["industry_group"] == group]
        ax.plot(temp["period"],
                temp["compensation"],
                label=group,
                linewidth=3)

    ax.set_title("Quarterly Compensation by Industry Group")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Compensation ($ in M)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollars_millions))
    ax.grid(True)
    ax.legend(ncols=2)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Dashboard layout
# -----------------------------------------------------------------------------

st.title("NYS Industry Compensation Forecast Dashboard")
st.caption(
    "A user-controlled tool to explore projected compensation growth for New York State's top industries."
)

with st.sidebar:
    st.header("Controls")
    data_file = "data.csv"

try:
    results = run_models(data_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

raw_df: pd.DataFrame = results["df"]  # type: ignore[assignment]
series_dict: dict[str, pd.Series] = results["series_dict"]  # type: ignore[assignment]
holdout_forecasts: dict[str, pd.DataFrame] = results["holdout_forecasts"]  # type: ignore[assignment]
future_forecasts: dict[str, pd.DataFrame] = results["future_forecasts"]  # type: ignore[assignment]
future_intervals: dict[str, dict[str, pd.DataFrame]] = results["future_intervals"]  # type: ignore[assignment]
residual_data: dict[str, pd.DataFrame] = results["residual_data"]  # type: ignore[assignment]
error_tables: dict[str, pd.DataFrame] = results["error_tables"]  # type: ignore[assignment]
ljung_box_tables: dict[str, pd.DataFrame] = results["ljung_box_tables"]  # type: ignore[assignment]

with st.sidebar:
    selected_display = st.selectbox("Industry", list(DISPLAY_TO_KEY.keys()), index=0)
    selected_key = DISPLAY_TO_KEY[selected_display]

    selected_models = st.multiselect(
        "Forecast models",
        options=["ETS", "ARIMA", "Random Forest"],
        default=["ETS", "ARIMA", "Random Forest"],
    )

    horizon = st.slider("Forecast horizon / quarters", 1, MAX_HORIZON, 8)

    selected_series = series_dict[selected_key]
    start_date = st.date_input(
        "Historical start date",
        value=selected_series.index[max(0, len(selected_series) - 40)].date(),
        min_value=selected_series.index.min().date(),
        max_value=selected_series.index.max().date(),
    )

    selected_resid_model = st.selectbox(
        "Residual diagnostic model",
        options=["ETS", "ARIMA", "Random Forest"],
        index=1,
    )

st.subheader("Overview")
st.write(
    '''Compensation levels are a critical economic indicator.
    They help inform the level of power that workers have in the labor marketplace relative to firms, the relative boom and bust across various industries, and general economic health.
    Economists and policy makers also monitor compensation levels for practical purposes, as they heavily influence state revenue collections in a given year. This project
    examined the top four largest industries within New York State according to BEA data. Users may view recent trends in compensation levels for these industries and compare forecasts,
    and their supporting analyses, below.
    '''
)

summary_cols = st.columns(4)
summary_cols[0].metric("Selected industry", selected_display)
latest_period = selected_series.index.max()
summary_cols[1].metric("Latest quarter", f"{latest_period.year} Q{latest_period.quarter}")
summary_cols[2].metric("Latest compensation", dollars_millions(float(selected_series.iloc[-1])))
best_model = error_tables[selected_key].index[0]
summary_cols[3].metric("Best holdout model by MAPE", best_model)

st.divider()

tab_forecast, tab_holdout, tab_diagnostics, tab_data = st.tabs(
    ["Forecasts", "Test Data Evaluation", "Residual Diagnostics", "Data"]
)

with tab_forecast:
    st.subheader("Select Time Series Model Forecasts")
    if not selected_models:
        st.warning("Select at least one forecast model in the sidebar.")
    else:
        fig = plot_actuals_and_forecast(
            historical=selected_series,
            future_fc=future_forecasts[selected_key],
            intervals=future_intervals[selected_key],
            models=selected_models,
            horizon=horizon,
            start_date=pd.Timestamp(start_date),
            title=f"{selected_display}: Actuals and {horizon}-quarter forecast",
        )
        st.pyplot(fig, clear_figure=True)

        st.caption(
            'Source: U.S. Bureau of Economic Analysis, "SQINC6N Compensation of employees by NAICS industry" (accessed Sunday, May 5, 2026)"'
        )
        st.dataframe(
            future_forecasts[selected_key][selected_models]
            .iloc[:horizon]
            .style.format("{:,.0f}"),
            use_container_width=True,
        )

with tab_holdout:
    st.subheader("Test Data Evaluation")
    st.write(
        f"Models were evaluated over the final {TEST_LENGTH} quarters of the available data. "
        "Metrics are calculated only against the test period.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        if selected_models:
            fig = plot_holdout_fit(
                holdout=holdout_forecasts[selected_key],
                models=selected_models,
                title=f"{selected_display}: holdout actuals vs forecasts",
            )
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Select at least one forecast model in the sidebar.")

    with col2:
        st.markdown("**Error metrics**")
        st.dataframe(style_metric_table(error_tables[selected_key]), use_container_width=True)

        st.markdown("**Ljung-Box residual check**")
        st.dataframe(
            style_metric_table(ljung_box_tables[selected_key]),
            use_container_width=True,
        )

with tab_diagnostics:
    st.subheader("Residual Diagnostics")
    residuals = residual_data[selected_key][selected_resid_model].dropna()
    fig = plot_residuals(
        residuals,
        title=f"{selected_display}: {selected_resid_model} holdout residual diagnostics",
    )
    st.pyplot(fig, clear_figure=True)

    lb = acorr_ljungbox(residuals, lags=[8, 12], return_df=True)
    lb = lb.rename(columns={"lb_stat": "Q Statistic", "lb_pvalue": "p value"})
    st.dataframe(lb.style.format({"Q Statistic": "{:.3f}", "p value": "{:.4f}"}), use_container_width=True)

with tab_data:
    st.subheader("Data Overview")
    fig = plot_industry_overview(raw_df)
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Cleaned data sample**")
    st.dataframe(
        raw_df[["period", "industry", "industry_group", "compensation"]]
        .sort_values(["period", "industry"])
        .tail(40),
        use_container_width=True,
    )

    st.markdown("**Target series coverage**")
    coverage = pd.DataFrame(
        [
            {
                "Series": SERIES_LABELS[key],
                "Start": s.index.min().date(),
                "End": s.index.max().date(),
                "Observations": len(s),
                "Missing values": int(s.isna().sum()),
            }
            for key, s in series_dict.items()
        ]
    )
    st.dataframe(coverage, use_container_width=True)
