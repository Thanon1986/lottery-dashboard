from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from predictor import (
    HYBRID_METHOD,
    METHOD,
    NOTE,
    PREDICTION_HISTORY_COLUMNS,
    SAFETY_NOTE,
    SUPPORTED_METHODS,
    WEIGHTED_METHOD,
    backtest_methods,
    load_history as predictor_load_history,
    phase3_analysis,
    read_prediction_history,
    rolling_accuracy_trend,
    run_backtest,
    save_analysis,
)


BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "data" / "lottery_history.csv"
SUMMARY_PATH = BASE_DIR / "output" / "stat_summary.xlsx"
ANALYZER_PATH = BASE_DIR / "analyzer.py"
PREDICTION_HISTORY_PATH = BASE_DIR / "output" / "prediction_history.csv"
BACKTEST_RESULT_PATH = BASE_DIR / "output" / "backtest_result.xlsx"
BACKTEST_DETAIL_PATH = BASE_DIR / "output" / "backtest_detail.csv"
DATA_QUALITY_REPORT_PATH = BASE_DIR / "output" / "data_quality_report.xlsx"

REQUIRED_HISTORY_COLUMNS = [
    "date",
    "first_prize",
    "last2",
    "front3_1",
    "front3_2",
    "back3_1",
    "back3_2",
]

NUMBER_COLUMNS = [
    "first_prize",
    "front3_1",
    "front3_2",
    "last2",
    "back3_1",
    "back3_2",
]

FIELD_LENGTHS = {
    "first_prize": 6,
    "last2": 2,
    "front3_1": 3,
    "front3_2": 3,
    "back3_1": 3,
    "back3_2": 3,
}

BACKUP_DIR = BASE_DIR / "backup"


st.set_page_config(
    page_title="Lottery History Dashboard",
    page_icon="",
    layout="wide",
)

st.sidebar.header("Settings")
selected_method = st.sidebar.selectbox("Prediction method", SUPPORTED_METHODS, index=SUPPORTED_METHODS.index(HYBRID_METHOD))
selected_rolling_span = st.sidebar.selectbox("Rolling span", [0, 20, 50, 100], index=0, format_func=lambda value: "All prior rows" if value == 0 else f"{value} rows")
selected_top_n = st.sidebar.slider("Top N suggestions", min_value=5, max_value=20, value=20, step=1)


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].astype("string").fillna("").str.strip()
    return result


@st.cache_data(show_spinner=False)
def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_HISTORY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    df = normalize_text_columns(df[REQUIRED_HISTORY_COLUMNS], REQUIRED_HISTORY_COLUMNS)
    df["date"] = df["date"].apply(normalize_date)
    for column, length in FIELD_LENGTHS.items():
        df[column] = df[column].apply(lambda value, col=column, size=length: normalize_number(value, size, col))
    df = df.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_stat_summary(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, engine="openpyxl")
    required_sheets = ["Summary", "Digit_Frequency", "Top_Last2", "Top_3Digit"]
    missing = [sheet for sheet in required_sheets if sheet not in sheets]
    if missing:
        raise ValueError(f"Missing required sheet(s): {', '.join(missing)}")
    return {name: normalize_text_columns(sheets[name], list(sheets[name].columns)) for name in required_sheets}


@st.cache_data(show_spinner=False)
def load_phase4_analysis(history_path: Path, top_n: int) -> dict:
    return phase3_analysis(history_path, top_n=top_n)


@st.cache_data(show_spinner=False)
def load_model_comparison(history_path: Path, top_n: int, rolling_span: int) -> dict:
    span = None if rolling_span == 0 else rolling_span
    return run_backtest(predictor_load_history(history_path), top_n=top_n, rolling_span=span)


@st.cache_data(show_spinner=False)
def load_prediction_history(path: Path) -> pd.DataFrame:
    rows = read_prediction_history(path)
    return pd.DataFrame(rows, columns=PREDICTION_HISTORY_COLUMNS).astype("string").fillna("")


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def show_file_error(message: str) -> None:
    st.error(message)
    st.stop()


def normalize_date(value: str) -> str:
    """Normalize supported dates to YYYY-MM-DD for in-app use."""
    raw_value = str(value).strip()
    formats = [
        ("%Y-%m-%d", "YYYY-MM-DD"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
    ]
    for date_format, _label in formats:
        try:
            return datetime.strptime(raw_value, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format {raw_value!r}; expected YYYY-MM-DD or DD/MM/YYYY")


def normalize_number(value: str, length: int, column: str) -> str:
    """Normalize fixed-width lottery values for display/analysis without editing CSV."""
    raw_value = str(value).strip()
    if not raw_value.isdigit():
        raise ValueError(f"Invalid {column} value {raw_value!r}; digits only")
    if len(raw_value) > length:
        raise ValueError(f"Invalid {column} value {raw_value!r}; max {length} digits")
    return raw_value.zfill(length)


def validate_new_lottery_row(row: dict[str, str], history: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    try:
        row["date"] = normalize_date(row["date"])
    except ValueError as exc:
        errors.append(str(exc))

    for field, length in FIELD_LENGTHS.items():
        value = row[field].strip()
        if not re.fullmatch(rf"\d{{{length}}}", value):
            errors.append(f"{field} must be exactly {length} digits.")

    if not errors and row["date"] in set(history["date"].apply(normalize_date)):
        errors.append(f"date already exists: {row['date']}")

    return errors


def backup_history_csv(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}__backup__{stamp}__streamlit_add_row{path.suffix}"
    shutil.copy2(path, backup_path)
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError(f"Backup failed: {backup_path}")
    return backup_path


def save_new_lottery_row(row: dict[str, str], history: pd.DataFrame) -> Path:
    backup_path = backup_history_csv(HISTORY_PATH)
    new_df = pd.concat([pd.DataFrame([row], columns=REQUIRED_HISTORY_COLUMNS), history], ignore_index=True)
    new_df = normalize_text_columns(new_df[REQUIRED_HISTORY_COLUMNS], REQUIRED_HISTORY_COLUMNS)
    new_df = new_df.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)
    temp_path = HISTORY_PATH.with_name(f"{HISTORY_PATH.stem}__streamlit_tmp{HISTORY_PATH.suffix}")
    new_df.to_csv(temp_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    shutil.move(str(temp_path), str(HISTORY_PATH))
    return backup_path


def short_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def recalculate_summary() -> tuple[bool, str]:
    if not ANALYZER_PATH.exists():
        return False, f"Missing analyzer.py: {ANALYZER_PATH}"
    try:
        result = subprocess.run(
            [sys.executable, str(ANALYZER_PATH)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        return False, short_traceback(exc)

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "analyzer.py failed without output").strip()
        if result.stdout:
            try:
                payload = json.loads(result.stdout)
                critical_errors = payload.get("critical_errors") or []
                if critical_errors:
                    details = "\n".join(str(error) for error in critical_errors)
            except json.JSONDecodeError:
                pass
        short_details = "\n".join(details.splitlines()[-12:])
        return False, short_details

    st.cache_data.clear()
    return True, "Summary recalculated successfully."


def run_backtest_export(top_n: int, rolling_span: int) -> tuple[bool, str, dict | None]:
    try:
        span = None if rolling_span == 0 else rolling_span
        result = backtest_methods(
            HISTORY_PATH,
            BACKTEST_RESULT_PATH,
            BACKUP_DIR,
            detail_csv_path=BACKTEST_DETAIL_PATH,
            top_n=top_n,
            rolling_span=span,
        )
        st.cache_data.clear()
        return True, f"Backtest completed: {BACKTEST_RESULT_PATH}", result
    except Exception as exc:
        return False, short_traceback(exc), None


st.title("Lottery History Dashboard")
st.warning(SAFETY_NOTE)

if not HISTORY_PATH.exists():
    show_file_error(f"Missing data file: {HISTORY_PATH}")
if not SUMMARY_PATH.exists():
    show_file_error(f"Missing summary file: {SUMMARY_PATH}")

try:
    history_df = load_history(HISTORY_PATH)
    summary_sheets = load_stat_summary(SUMMARY_PATH)
    phase3 = load_phase4_analysis(HISTORY_PATH, selected_top_n)
    model_comparison = load_model_comparison(HISTORY_PATH, selected_top_n, selected_rolling_span)
    prediction_history_df = load_prediction_history(PREDICTION_HISTORY_PATH)
except Exception as exc:
    show_file_error(str(exc))

summary_df = summary_sheets["Summary"]
digit_df = summary_sheets["Digit_Frequency"]
top_last2_df = summary_sheets["Top_Last2"]
top_3digit_df = summary_sheets["Top_3Digit"]

for column in ["total", "first_prize", "front3_1", "front3_2", "last2", "back3_1", "back3_2"]:
    if column in digit_df.columns:
        digit_df[column] = numeric_column(digit_df, column)

for frame in [top_last2_df, top_3digit_df]:
    for column in ["rank", "count", "front3_1", "front3_2", "back3_1", "back3_2"]:
        if column in frame.columns:
            frame[column] = numeric_column(frame, column).astype(int)
    if "share" in frame.columns:
        frame["share"] = numeric_column(frame, "share")

metric_cols = st.columns(4)
metric_cols[0].metric("History Rows", f"{len(history_df):,}")
metric_cols[1].metric("Latest Draw", history_df["date"].iloc[0])
metric_cols[2].metric("Current Method", selected_method)
metric_cols[3].metric("Last Updated", phase3["last_updated"])

tab_history, tab_add, tab_next, tab_prediction_history, tab_model, tab_backtest, tab_summary, tab_digits, tab_last2, tab_3digit = st.tabs(
    [
        "Lottery History",
        "Add Result",
        "Next Draw Analysis",
        "Prediction History",
        "Model Comparison",
        "Backtest",
        "Summary",
        "Digit Frequency",
        "Top Last2",
        "Top 3Digit",
    ]
)

with tab_history:
    st.subheader("Lottery History")
    st.info(NOTE)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

with tab_add:
    st.subheader("Add New Lottery Result")
    st.info(NOTE)
    success_message = st.session_state.pop("add_result_success", None)
    if success_message:
        st.success(success_message)
    with st.form("add_lottery_result", clear_on_submit=False):
        date = st.text_input("date", placeholder="YYYY-MM-DD or DD/MM/YYYY", max_chars=10)
        form_cols = st.columns(3)
        first_prize = form_cols[0].text_input("first_prize", max_chars=6)
        last2 = form_cols[1].text_input("last2", max_chars=2)
        front3_1 = form_cols[2].text_input("front3_1", max_chars=3)
        front3_2 = form_cols[0].text_input("front3_2", max_chars=3)
        back3_1 = form_cols[1].text_input("back3_1", max_chars=3)
        back3_2 = form_cols[2].text_input("back3_2", max_chars=3)
        submitted = st.form_submit_button("Save result")

    if submitted:
        new_row = {
            "date": date.strip(),
            "first_prize": first_prize.strip(),
            "last2": last2.strip(),
            "front3_1": front3_1.strip(),
            "front3_2": front3_2.strip(),
            "back3_1": back3_1.strip(),
            "back3_2": back3_2.strip(),
        }
        validation_errors = validate_new_lottery_row(new_row, history_df)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            try:
                backup_path = save_new_lottery_row(new_row, history_df)
                summary_ok, summary_message = recalculate_summary()
                backtest_ok, backtest_message, _result = run_backtest_export(selected_top_n, selected_rolling_span)
                if summary_ok and backtest_ok:
                    st.session_state["add_result_success"] = (
                        f"Saved successfully. Backup created: {backup_path}. "
                        "Summary, scoring, and backtest were refreshed."
                    )
                else:
                    st.session_state["add_result_success"] = (
                        f"Saved successfully. Backup created: {backup_path}. "
                        f"Refresh messages: {summary_message}; {backtest_message}"
                    )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")

with tab_next:
    st.subheader("Next Draw Analysis")
    st.info(NOTE)
    st.caption(f"Source rows: {phase3['source_rows']}")

    suggested_last2_df = pd.DataFrame(phase3["frequency"]["suggested_last2"]).astype("string").fillna("")
    suggested_3digit_df = pd.DataFrame(phase3["frequency"]["suggested_3digit"]).astype("string").fillna("")
    weighted_last2_df = pd.DataFrame(phase3["weighted_recent"]["suggested_last2"]).astype("string").fillna("")
    weighted_3digit_df = pd.DataFrame(phase3["weighted_recent"]["suggested_3digit"]).astype("string").fillna("")
    hybrid_last2_df = pd.DataFrame(phase3["hybrid"]["suggested_last2"][:selected_top_n]).astype("string").fillna("")
    hybrid_3digit_df = pd.DataFrame(phase3["hybrid"]["suggested_3digit"][:selected_top_n]).astype("string").fillna("")

    st.markdown(f"**{METHOD}**")
    freq_cols = st.columns(2)
    with freq_cols[0]:
        st.subheader(f"Frequency Last2 Top {selected_top_n}")
        st.dataframe(suggested_last2_df, use_container_width=True, hide_index=True)
    with freq_cols[1]:
        st.subheader(f"Frequency 3Digit Top {selected_top_n}")
        st.dataframe(suggested_3digit_df, use_container_width=True, hide_index=True)

    st.markdown(f"**{WEIGHTED_METHOD}**")
    weighted_cols = st.columns(2)
    with weighted_cols[0]:
        st.subheader(f"Weighted Last2 Top {selected_top_n}")
        st.dataframe(weighted_last2_df, use_container_width=True, hide_index=True)
    with weighted_cols[1]:
        st.subheader(f"Weighted 3Digit Top {selected_top_n}")
        st.dataframe(weighted_3digit_df, use_container_width=True, hide_index=True)

    st.markdown(f"**{HYBRID_METHOD}**")
    hybrid_cols = st.columns(2)
    with hybrid_cols[0]:
        st.subheader(f"Hybrid Last2 Top {selected_top_n}")
        st.dataframe(hybrid_last2_df, use_container_width=True, hide_index=True)
    with hybrid_cols[1]:
        st.subheader(f"Hybrid 3Digit Top {selected_top_n}")
        st.dataframe(hybrid_3digit_df, use_container_width=True, hide_index=True)

    st.subheader("Digit Position Analysis")
    position_cols = st.columns(2)
    with position_cols[0]:
        st.caption("last2 positions")
        st.dataframe(pd.DataFrame(phase3["positions"]["last2"]).astype("string"), use_container_width=True, hide_index=True)
    with position_cols[1]:
        st.caption("3digit positions")
        st.dataframe(pd.DataFrame(phase3["positions"]["three_digit"]).astype("string"), use_container_width=True, hide_index=True)

    st.subheader("Hot / Cold Numbers")
    hot_cold = phase3["hot_cold"]
    hot_cols = st.columns(4)
    with hot_cols[0]:
        st.caption("Hot last2")
        st.dataframe(pd.DataFrame(hot_cold["hot_last2"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[1]:
        st.caption("Hot 3digit")
        st.dataframe(pd.DataFrame(hot_cold["hot_3digit"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[2]:
        st.caption("Cold last2")
        st.dataframe(pd.DataFrame(hot_cold["cold_last2"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[3]:
        st.caption("Cold 3digit")
        st.dataframe(pd.DataFrame(hot_cold["cold_3digit"]).astype("string"), use_container_width=True, hide_index=True)

    with st.form("save_next_draw_analysis", clear_on_submit=False):
        target_draw = st.text_input("target_draw", placeholder="YYYY-MM-DD or DD/MM/YYYY", max_chars=10)
        method_choice = st.selectbox("method", SUPPORTED_METHODS, index=SUPPORTED_METHODS.index(selected_method))
        save_submitted = st.form_submit_button("Save Analysis")

    save_message = st.session_state.pop("save_analysis_success", None)
    if save_message:
        st.success(save_message)

    if save_submitted:
        try:
            selected_suggestions = {
                METHOD: phase3["frequency"],
                WEIGHTED_METHOD: phase3["weighted_recent"],
                HYBRID_METHOD: phase3["hybrid"],
            }[method_choice]
            backup_path = save_analysis(
                PREDICTION_HISTORY_PATH,
                BACKUP_DIR,
                target_draw,
                selected_suggestions,
            )
            st.cache_data.clear()
            if backup_path:
                st.session_state["save_analysis_success"] = f"Analysis saved. Backup created: {backup_path}"
            else:
                st.session_state["save_analysis_success"] = "Analysis saved. New prediction history file created."
            st.rerun()
        except Exception as exc:
            st.error(f"Save Analysis failed: {exc}")

    st.caption(NOTE)

with tab_prediction_history:
    st.subheader("Prediction History")
    st.info(NOTE)
    st.dataframe(prediction_history_df, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Model Comparison")
    st.info(NOTE)
    comparison_df = pd.DataFrame(model_comparison["summary"])
    trend_df = pd.DataFrame(rolling_accuracy_trend(model_comparison["details"]))
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    hit_rate_df = comparison_df[["method", "last2_hit_rate", "3digit_hit_rate", "recent_20_round_hit_rate"]].copy()
    hit_rate_df = hit_rate_df.set_index("method")
    st.subheader("Hit Rate Comparison")
    st.plotly_chart(px.bar(hit_rate_df, barmode="group"), use_container_width=True)
    if not trend_df.empty:
        st.subheader("Rolling Accuracy Trend")
        chart_df = trend_df.pivot(index="draw_date", columns="method", values="rolling_hit_rate")
        st.plotly_chart(px.line(chart_df), use_container_width=True)

with tab_backtest:
    st.subheader("Backtest")
    st.info(NOTE)
    st.write("Backtest uses only rows before each tested draw.")
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            ok, message, result = run_backtest_export(selected_top_n, selected_rolling_span)
        if ok and result:
            st.success(message)
            st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True, hide_index=True)
        else:
            st.error("Backtest failed.")
            st.code(message, language="text")
    if BACKTEST_RESULT_PATH.exists():
        st.caption(f"Last result file: {BACKTEST_RESULT_PATH}")
    if BACKTEST_DETAIL_PATH.exists():
        st.caption(f"Detail CSV: {BACKTEST_DETAIL_PATH}")

with tab_summary:
    st.subheader("Summary")
    st.info(NOTE)
    recalc_success = st.session_state.pop("recalculate_success", None)
    if recalc_success:
        st.success(recalc_success)
    if st.button("Recalculate Summary", type="primary"):
        with st.spinner("Recalculating summary..."):
            ok, message = recalculate_summary()
        if ok:
            st.session_state["recalculate_success"] = message
            st.rerun()
        else:
            st.error("Recalculate Summary failed.")
            st.code(message, language="text")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab_digits:
    st.subheader("Digit Frequency")
    st.info(NOTE)
    chart_df = digit_df[["digit", "total"]].copy()
    chart_df["digit"] = chart_df["digit"].astype(str)
    chart_df = chart_df.set_index("digit")
    st.plotly_chart(px.bar(chart_df, y="total"), use_container_width=True)
    st.dataframe(digit_df, use_container_width=True, hide_index=True)

with tab_last2:
    st.subheader("Top 20 Last 2 Digits")
    st.info(NOTE)
    st.dataframe(top_last2_df, use_container_width=True, hide_index=True)

with tab_3digit:
    st.subheader("Top 20 Three-Digit Numbers")
    st.info(NOTE)
    st.dataframe(top_3digit_df, use_container_width=True, hide_index=True)
