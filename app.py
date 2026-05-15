from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from localization import DEFAULT_LANGUAGE, LANGUAGE_LABELS, LANGUAGE_OPTIONS, TEXT, get_text

st.set_page_config(
    page_title=TEXT[DEFAULT_LANGUAGE]["app_title"],
    page_icon="",
    layout="wide",
)

if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE


def current_language() -> str:
    language = st.session_state.get("language", DEFAULT_LANGUAGE)
    return language if language in LANGUAGE_OPTIONS else DEFAULT_LANGUAGE


def tr(key: str) -> str:
    return get_text(current_language(), key)


def render_language_selector(key: str = "language_selector", label_visibility: str = "visible") -> None:
    language_labels = [LANGUAGE_LABELS[language] for language in LANGUAGE_OPTIONS]
    selected_language_label = st.selectbox(
        tr("language"),
        language_labels,
        index=LANGUAGE_OPTIONS.index(current_language()),
        key=key,
        label_visibility=label_visibility,
    )
    selected_language = next(language for language, label in LANGUAGE_LABELS.items() if label == selected_language_label)
    if selected_language != current_language():
        st.session_state["language"] = selected_language
        st.rerun()


def read_login_config() -> tuple[str, str]:
    try:
        username = st.secrets["auth"]["username"]
        password = st.secrets["auth"]["password"]
    except Exception:
        st.error(tr("missing_login_config"))
        st.stop()
    return str(username), str(password)


def require_login() -> None:
    with st.sidebar:
        render_language_selector(key="sidebar_language_selector")

    app_username, app_password = read_login_config()
    is_authenticated = (
        st.session_state.get("authenticated") is True
        and st.session_state.get("authenticated_username") == app_username
    )

    if is_authenticated:
        with st.sidebar:
            st.success(tr("logged_in"))
            if st.button(tr("logout")):
                st.session_state.pop("authenticated", None)
                st.session_state.pop("authenticated_username", None)
                st.rerun()
        return

    st.title(tr("login"))
    with st.form("login_form"):
        username = st.text_input(tr("username"))
        password = st.text_input(tr("password"), type="password")
        submitted = st.form_submit_button(tr("login"))

    if submitted:
        if username == app_username and password == app_password:
            st.session_state["authenticated"] = True
            st.session_state["authenticated_username"] = app_username
            st.rerun()
        st.error(tr("invalid_login"))
    st.stop()


require_login()

from predictor import (  # noqa: E402 - imported only after login gate blocks unauthenticated users.
    HYBRID_METHOD,
    METHOD,
    MODEL_ACCURACY_COLUMNS,
    PREDICTION_HISTORY_COLUMNS,
    SUPPORTED_METHODS,
    WEIGHTED_METHOD,
    backtest_methods,
    load_history as predictor_load_history,
    phase3_analysis,
    refresh_model_accuracy,
    read_prediction_history,
    rolling_accuracy_trend,
    run_backtest,
    save_analysis,
    save_model_accuracy_prediction,
)
from config import APP_CONFIG  # noqa: E402
from constants import (  # noqa: E402
    ANALYZER_PATH,
    BACKTEST_DETAIL_PATH,
    BACKTEST_RESULT_PATH,
    BACKUP_DIR,
    BACKUP_RETENTION_LIMIT,
    BACKUP_TARGETS,
    BASE_DIR,
    EXPORT_DIR,
    FIELD_LENGTHS,
    HISTORY_PATH,
    LATEST_PREDICTION_PATH,
    MODEL_ACCURACY_PATH,
    PREDICTION_HISTORY_PATH,
    REQUIRED_HISTORY_COLUMNS,
    SUMMARY_PATH,
)
from utils import (  # noqa: E402
    accuracy_leaderboard,
    append_system_log,
    build_ai_insights,
    create_csv_backup,
    data_quality_status,
    ensure_operational_dirs,
    export_csv_file,
    export_full_system_zip,
    folder_size_mb,
    generate_latest_prediction_csv,
    latest_export_time,
    list_backup_files,
    load_system_log,
    normalize_date,
    normalize_number,
    normalize_text_columns,
    numeric_column,
    read_insight_history,
    restore_backup_file,
    restore_history_from_backup,
    safe_read_csv,
    save_insight_history_if_new,
    save_new_lottery_row,
    short_traceback,
    validate_csv_for_target,
    validate_new_lottery_row,
)


st.sidebar.header(tr("settings"))
selected_method = st.sidebar.selectbox(tr("prediction_method"), SUPPORTED_METHODS, index=SUPPORTED_METHODS.index(HYBRID_METHOD))
selected_rolling_span = st.sidebar.selectbox(
    tr("rolling_span"),
    list(APP_CONFIG.rolling_spans),
    index=0,
    format_func=lambda value: tr("all_prior_rows") if value == 0 else f"{value} {tr('rows')}",
)
selected_top_n = st.sidebar.slider(
    tr("top_n_suggestions"),
    min_value=APP_CONFIG.min_top_n,
    max_value=APP_CONFIG.max_top_n,
    value=APP_CONFIG.default_top_n,
    step=APP_CONFIG.top_n_step,
)

@st.cache_data(show_spinner=False)
def load_history(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path, REQUIRED_HISTORY_COLUMNS)
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


def load_model_accuracy() -> tuple[pd.DataFrame, dict]:
    result = refresh_model_accuracy(HISTORY_PATH, MODEL_ACCURACY_PATH, BACKUP_DIR)
    rows = result["rows"]
    df = pd.DataFrame(rows, columns=MODEL_ACCURACY_COLUMNS).astype("string").fillna("")
    return df, result


def show_file_error(message: str) -> None:
    st.error(message)
    st.stop()


def recalculate_summary() -> tuple[bool, str]:
    if not ANALYZER_PATH.exists():
        return False, f"Missing analyzer.py: {ANALYZER_PATH}"
    try:
        result = subprocess.run(
            [sys.executable, str(ANALYZER_PATH)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=APP_CONFIG.subprocess_timeout_seconds,
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
    return True, tr("summary_recalculated")


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
        return True, f"{tr('backtest_completed')}: {BACKTEST_RESULT_PATH}", result
    except Exception as exc:
        return False, short_traceback(exc), None


def run_auto_refresh(top_n: int, rolling_span: int) -> list[str]:
    messages: list[str] = []

    append_system_log("refresh_statistics", "started", "Running analyzer.py for summary and frequency tables.")
    summary_ok, summary_message = recalculate_summary()
    if not summary_ok:
        append_system_log("refresh_statistics", "failed", summary_message)
        raise RuntimeError(summary_message)
    append_system_log("refresh_statistics", "success", summary_message)
    messages.append(summary_message)

    append_system_log("refresh_model_accuracy", "started", "Comparing saved predictions with available actual results.")
    accuracy_backup = create_csv_backup(MODEL_ACCURACY_PATH)
    accuracy_result = refresh_model_accuracy(HISTORY_PATH, MODEL_ACCURACY_PATH, BACKUP_DIR)
    append_system_log(
        "refresh_model_accuracy",
        "success",
        f"evaluated={accuracy_result['evaluated_predictions']}; total={accuracy_result['total_predictions']}; backup={accuracy_backup or ''}",
    )
    messages.append(tr("accuracy_leaderboard_refreshed"))

    append_system_log("refresh_prediction_history", "started", "Validating prediction history file.")
    prediction_rows = read_prediction_history(PREDICTION_HISTORY_PATH)
    append_system_log("refresh_prediction_history", "success", f"records={len(prediction_rows)}")
    messages.append(tr("prediction_history_refreshed"))

    append_system_log("generate_latest_prediction", "started", "Generating latest statistical suggestions.")
    latest_backup = generate_latest_prediction_csv(top_n)
    latest_detail = f"latest_prediction={LATEST_PREDICTION_PATH}"
    if latest_backup:
        latest_detail += f"; backup={latest_backup}"
    append_system_log("generate_latest_prediction", "success", latest_detail)
    messages.append(tr("latest_prediction_generated"))

    append_system_log("refresh_backtest", "started", "Running rolling backtest and model comparison outputs.")
    backtest_ok, backtest_message, _result = run_backtest_export(top_n, rolling_span)
    if not backtest_ok:
        append_system_log("refresh_backtest", "failed", backtest_message)
        raise RuntimeError(backtest_message)
    append_system_log("refresh_backtest", "success", backtest_message)
    messages.append(tr("backtest_model_refreshed"))

    st.cache_data.clear()
    return messages


def process_new_result(row: dict[str, str], history: pd.DataFrame, top_n: int, rolling_span: int) -> list[str]:
    backup_path: Path | None = None
    try:
        append_system_log("add_result", "started", f"draw_date={row['date']}")
        backup_path = save_new_lottery_row(row, history)
        append_system_log("add_result", "success", f"draw_date={row['date']}; backup={backup_path}")
        messages = [f"{tr('saved_result_backup')}: {backup_path}"]
        messages.extend(run_auto_refresh(top_n, rolling_span))
        append_system_log("auto_result_processing", "success", f"draw_date={row['date']}")
        return messages
    except Exception as exc:
        detail = short_traceback(exc)
        append_system_log("auto_result_processing", "failed", detail)
        if backup_path is not None:
            try:
                restore_history_from_backup(backup_path)
                append_system_log("rollback", "success", f"Restored lottery history from {backup_path}")
            except Exception as rollback_exc:
                rollback_detail = short_traceback(rollback_exc)
                append_system_log("rollback", "failed", rollback_detail)
                raise RuntimeError(f"{detail}; rollback failed: {rollback_detail}") from rollback_exc
        raise RuntimeError(detail) from exc


render_language_selector(key="main_language_selector")
st.title(tr("app_title"))
st.warning(tr("safety_note"))

if not HISTORY_PATH.exists():
    show_file_error(f"{tr('missing_data_file')}: {HISTORY_PATH}")
if not SUMMARY_PATH.exists():
    show_file_error(f"{tr('missing_summary_file')}: {SUMMARY_PATH}")

try:
    history_df = load_history(HISTORY_PATH)
    summary_sheets = load_stat_summary(SUMMARY_PATH)
    phase3 = load_phase4_analysis(HISTORY_PATH, selected_top_n)
    model_comparison = load_model_comparison(HISTORY_PATH, selected_top_n, selected_rolling_span)
    prediction_history_df = load_prediction_history(PREDICTION_HISTORY_PATH)
    model_accuracy_df, model_accuracy_result = load_model_accuracy()
    ai_insight_result = build_ai_insights(history_df)
    insight_save_result = save_insight_history_if_new(ai_insight_result)
    insight_history_df = read_insight_history()
    system_log_df = load_system_log(20)
    backup_df = list_backup_files()
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
metric_cols[0].metric(tr("history_rows"), f"{len(history_df):,}")
metric_cols[1].metric(tr("latest_draw"), history_df["date"].iloc[0])
metric_cols[2].metric(tr("current_method"), selected_method)
metric_cols[3].metric(tr("last_updated"), phase3["last_updated"])

tab_history, tab_add, tab_next, tab_prediction_history, tab_accuracy, tab_insights, tab_status, tab_backup, tab_model, tab_backtest, tab_summary, tab_digits, tab_last2, tab_3digit = st.tabs(
    [
        tr("tab_history"),
        tr("tab_add"),
        tr("tab_next"),
        tr("tab_prediction_history"),
        tr("tab_accuracy"),
        tr("tab_insights"),
        tr("tab_status"),
        tr("tab_backup"),
        tr("tab_model"),
        tr("tab_backtest"),
        tr("tab_summary"),
        tr("tab_digits"),
        tr("tab_last2"),
        tr("tab_3digit"),
    ]
)

with tab_history:
    st.subheader(tr("tab_history"))
    st.info(tr("safety_note"))
    st.dataframe(history_df, use_container_width=True, hide_index=True)

with tab_add:
    st.subheader(tr("add_new_result"))
    st.info(tr("safety_note"))
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
        submitted = st.form_submit_button(tr("save_result"))

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
                process_messages = process_new_result(new_row, history_df, selected_top_n, selected_rolling_span)
                st.session_state["add_result_success"] = f"{tr('auto_processing_completed')} " + " ".join(process_messages)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"{tr('auto_processing_failed')}: {exc}")

with tab_next:
    st.subheader(tr("next_draw_analysis"))
    st.info(tr("safety_note"))
    st.caption(f"{tr('source_rows')}: {phase3['source_rows']}")

    suggested_last2_df = pd.DataFrame(phase3["frequency"]["suggested_last2"]).astype("string").fillna("")
    suggested_3digit_df = pd.DataFrame(phase3["frequency"]["suggested_3digit"]).astype("string").fillna("")
    weighted_last2_df = pd.DataFrame(phase3["weighted_recent"]["suggested_last2"]).astype("string").fillna("")
    weighted_3digit_df = pd.DataFrame(phase3["weighted_recent"]["suggested_3digit"]).astype("string").fillna("")
    hybrid_last2_df = pd.DataFrame(phase3["hybrid"]["suggested_last2"][:selected_top_n]).astype("string").fillna("")
    hybrid_3digit_df = pd.DataFrame(phase3["hybrid"]["suggested_3digit"][:selected_top_n]).astype("string").fillna("")

    st.markdown(f"**{METHOD}**")
    freq_cols = st.columns(2)
    with freq_cols[0]:
        st.subheader(f"{tr('frequency_last2_top')} {selected_top_n}")
        st.dataframe(suggested_last2_df, use_container_width=True, hide_index=True)
    with freq_cols[1]:
        st.subheader(f"{tr('frequency_3digit_top')} {selected_top_n}")
        st.dataframe(suggested_3digit_df, use_container_width=True, hide_index=True)

    st.markdown(f"**{WEIGHTED_METHOD}**")
    weighted_cols = st.columns(2)
    with weighted_cols[0]:
        st.subheader(f"{tr('weighted_last2_top')} {selected_top_n}")
        st.dataframe(weighted_last2_df, use_container_width=True, hide_index=True)
    with weighted_cols[1]:
        st.subheader(f"{tr('weighted_3digit_top')} {selected_top_n}")
        st.dataframe(weighted_3digit_df, use_container_width=True, hide_index=True)

    st.markdown(f"**{HYBRID_METHOD}**")
    hybrid_cols = st.columns(2)
    with hybrid_cols[0]:
        st.subheader(f"{tr('hybrid_last2_top')} {selected_top_n}")
        st.dataframe(hybrid_last2_df, use_container_width=True, hide_index=True)
    with hybrid_cols[1]:
        st.subheader(f"{tr('hybrid_3digit_top')} {selected_top_n}")
        st.dataframe(hybrid_3digit_df, use_container_width=True, hide_index=True)

    st.subheader(tr("digit_position_analysis"))
    position_cols = st.columns(2)
    with position_cols[0]:
        st.caption(tr("last2_positions"))
        st.dataframe(pd.DataFrame(phase3["positions"]["last2"]).astype("string"), use_container_width=True, hide_index=True)
    with position_cols[1]:
        st.caption(tr("three_digit_positions"))
        st.dataframe(pd.DataFrame(phase3["positions"]["three_digit"]).astype("string"), use_container_width=True, hide_index=True)

    st.subheader(tr("hot_cold_numbers"))
    hot_cold = phase3["hot_cold"]
    hot_cols = st.columns(4)
    with hot_cols[0]:
        st.caption(tr("hot_last2"))
        st.dataframe(pd.DataFrame(hot_cold["hot_last2"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[1]:
        st.caption(tr("hot_3digit"))
        st.dataframe(pd.DataFrame(hot_cold["hot_3digit"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[2]:
        st.caption(tr("cold_last2"))
        st.dataframe(pd.DataFrame(hot_cold["cold_last2"]).astype("string"), use_container_width=True, hide_index=True)
    with hot_cols[3]:
        st.caption(tr("cold_3digit"))
        st.dataframe(pd.DataFrame(hot_cold["cold_3digit"]).astype("string"), use_container_width=True, hide_index=True)

    with st.form("save_next_draw_analysis", clear_on_submit=False):
        target_draw = st.text_input("target_draw", placeholder="YYYY-MM-DD or DD/MM/YYYY", max_chars=10)
        method_choice = st.selectbox(tr("method"), SUPPORTED_METHODS, index=SUPPORTED_METHODS.index(selected_method))
        save_submitted = st.form_submit_button(tr("save_analysis"))

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
            center_prediction_backup = create_csv_backup(PREDICTION_HISTORY_PATH)
            center_accuracy_backup = create_csv_backup(MODEL_ACCURACY_PATH)
            backup_path = save_analysis(
                PREDICTION_HISTORY_PATH,
                BACKUP_DIR,
                target_draw,
                selected_suggestions,
            )
            accuracy_backup_path = save_model_accuracy_prediction(
                MODEL_ACCURACY_PATH,
                BACKUP_DIR,
                target_draw,
                selected_suggestions,
            )
            st.cache_data.clear()
            backup_notes = [
                str(path)
                for path in [center_prediction_backup, center_accuracy_backup, backup_path, accuracy_backup_path]
                if path
            ]
            if backup_notes:
                st.session_state["save_analysis_success"] = f"{tr('analysis_saved_backup')}: {', '.join(backup_notes)}"
            else:
                st.session_state["save_analysis_success"] = tr("analysis_saved_tracking")
            st.rerun()
        except Exception as exc:
            st.error(f"{tr('save_analysis_failed')}: {exc}")

    st.caption(tr("safety_note"))

with tab_prediction_history:
    st.subheader(tr("prediction_history"))
    st.info(tr("safety_note"))
    st.dataframe(prediction_history_df, use_container_width=True, hide_index=True)

with tab_accuracy:
    st.subheader(tr("model_accuracy"))
    st.info(tr("safety_note"))

    total_predictions = len(model_accuracy_df)
    accuracy_work = model_accuracy_df.copy()
    for column in ["hit_last2", "hit_3digit", "score"]:
        accuracy_work[f"{column}_num"] = pd.to_numeric(accuracy_work[column], errors="coerce")
    evaluated_df = accuracy_work[accuracy_work["score_num"].notna()].copy()
    evaluated_count = len(evaluated_df)
    total_hit_units = int(evaluated_df["hit_last2_num"].sum() + evaluated_df["hit_3digit_num"].sum()) if evaluated_count else 0
    overall_hit_rate = total_hit_units / (evaluated_count * 2) if evaluated_count else 0

    accuracy_cols = st.columns(4)
    accuracy_cols[0].metric(tr("total_predictions"), f"{total_predictions:,}")
    accuracy_cols[1].metric(tr("evaluated"), f"{evaluated_count:,}")
    accuracy_cols[2].metric(tr("hit_rate_percent"), f"{overall_hit_rate:.2%}")
    accuracy_cols[3].metric(tr("tracking_file"), MODEL_ACCURACY_PATH.name)

    if model_accuracy_result.get("updated"):
        st.success(tr("accuracy_refreshed"))
    if model_accuracy_result.get("backup_path"):
        st.caption(f"{tr('backup_created')}: {model_accuracy_result['backup_path']}")

    st.subheader(tr("accuracy_table"))
    st.dataframe(model_accuracy_df, use_container_width=True, hide_index=True)

    if evaluated_df.empty:
        st.info(tr("no_evaluated_predictions"))
    else:
        leaderboard = (
            evaluated_df.groupby("method", dropna=False)
            .agg(
                total_predictions=("method", "size"),
                last2_hit_count=("hit_last2_num", "sum"),
                three_digit_hit_count=("hit_3digit_num", "sum"),
                total_score=("score_num", "sum"),
            )
            .reset_index()
        )
        leaderboard["last2_hit_rate"] = leaderboard["last2_hit_count"] / leaderboard["total_predictions"]
        leaderboard["3digit_hit_rate"] = leaderboard["three_digit_hit_count"] / leaderboard["total_predictions"]
        leaderboard["overall_hit_rate"] = leaderboard["total_score"] / (leaderboard["total_predictions"] * 2)
        leaderboard["avg_score"] = leaderboard["total_score"] / leaderboard["total_predictions"]
        leaderboard = leaderboard.sort_values(
            ["overall_hit_rate", "avg_score", "total_predictions", "method"],
            ascending=[False, False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

        best_method = str(leaderboard.iloc[0]["method"])
        st.success(f"{tr('best_current_model')}: {best_method}")

        def highlight_best(row: pd.Series) -> list[str]:
            return ["background-color: #d9f7be" if row["rank"] == 1 else "" for _value in row]

        st.subheader(tr("leaderboard_model_ranking"))
        st.dataframe(leaderboard.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

        rate_chart_df = leaderboard[["method", "last2_hit_rate", "3digit_hit_rate", "overall_hit_rate"]].melt(
            id_vars="method",
            var_name="metric",
            value_name="hit_rate",
        )
        st.subheader(tr("hit_rate_comparison"))
        st.plotly_chart(px.bar(rate_chart_df, x="method", y="hit_rate", color="metric", barmode="group"), use_container_width=True)

        trend_df = evaluated_df.sort_values(["method", "draw_date", "created_at"], kind="stable").copy()
        trend_df["accuracy_point"] = trend_df["score_num"] / 2
        trend_df["accuracy_trend"] = (
            trend_df.groupby("method")["accuracy_point"].expanding().mean().reset_index(level=0, drop=True)
        )
        st.subheader(tr("accuracy_trend"))
        st.plotly_chart(
            px.line(trend_df, x="draw_date", y="accuracy_trend", color="method", markers=True),
            use_container_width=True,
        )

with tab_insights:
    st.subheader(tr("ai_insights"))
    st.info(tr("safety_note"))
    st.caption(tr("insight_caption"))

    if insight_save_result.get("updated"):
        st.success(f"{tr('insight_history_updated')}: {insight_save_result['added']} {tr('new_cards')}")
    if insight_save_result.get("backup_path"):
        st.caption(f"{tr('insight_backup_created')}: {insight_save_result['backup_path']}")

    insight_cards = pd.DataFrame(ai_insight_result["insights"])
    if not insight_cards.empty:
        insight_cards["confidence_num"] = pd.to_numeric(insight_cards["confidence"], errors="coerce").fillna(0)
        avg_confidence = insight_cards["confidence_num"].mean()
        high_count = int((insight_cards["signal"] == "HIGH").sum())
        insight_metrics = st.columns(4)
        insight_metrics[0].metric(tr("insight_cards"), f"{len(insight_cards):,}")
        insight_metrics[1].metric(tr("average_confidence"), f"{avg_confidence:.0f}/100")
        insight_metrics[2].metric(tr("high_signals"), f"{high_count:,}")
        insight_metrics[3].metric(tr("generated"), str(ai_insight_result["generated_at"]))

        warning_list = ai_insight_result.get("warnings", [])
        if warning_list:
            st.warning(f"{tr('warnings')}: " + ", ".join(str(item) for item in warning_list))
        else:
            st.success(tr("no_insight_warnings"))

        for row_start in range(0, len(insight_cards), 2):
            card_cols = st.columns(2)
            for card_col, (_index, card) in zip(card_cols, insight_cards.iloc[row_start : row_start + 2].iterrows()):
                with card_col:
                    signal = str(card["signal"])
                    title = str(card["title"])
                    confidence = str(card["confidence"])
                    explanation = str(card["explanation"])
                    warning_text = str(card["warnings"])
                    st.markdown(f"**{title}**")
                    st.metric(tr("signal_confidence"), f"{signal}", f"{confidence}/100")
                    st.write(explanation)
                    if warning_text:
                        st.caption(f"{tr('warning')}: {warning_text}")
                    st.caption(f"{tr('generated')}: {card['generated_at']}")

    st.subheader(tr("insight_table"))
    st.dataframe(insight_cards.drop(columns=["confidence_num"], errors="ignore"), use_container_width=True, hide_index=True)

    st.subheader(tr("digit_heatmap"))
    heatmap_df = ai_insight_result["heatmap"]
    if isinstance(heatmap_df, pd.DataFrame) and not heatmap_df.empty:
        heatmap_pivot = heatmap_df.pivot(index="field", columns="digit", values="count")
        st.plotly_chart(px.imshow(heatmap_pivot, aspect="auto", color_continuous_scale="Blues"), use_container_width=True)

    st.subheader(tr("trend_chart"))
    trend_chart = ai_insight_result["trend"]
    if isinstance(trend_chart, pd.DataFrame) and not trend_chart.empty:
        st.plotly_chart(
            px.line(trend_chart, x="date", y="cumulative_count", color="last2", markers=False),
            use_container_width=True,
        )

    st.subheader(tr("digit_movement_chart"))
    movement_chart = ai_insight_result["movement"]
    if isinstance(movement_chart, pd.DataFrame) and not movement_chart.empty:
        st.plotly_chart(
            px.line(movement_chart, x="period", y="share", color="digit", markers=True),
            use_container_width=True,
        )

    st.subheader(tr("insight_history"))
    st.dataframe(insight_history_df.sort_values("generated_at", ascending=False, kind="stable"), use_container_width=True, hide_index=True)

with tab_status:
    st.subheader(tr("system_status"))
    st.info(tr("safety_note"))
    leaderboard_df, latest_accuracy, best_model = accuracy_leaderboard(model_accuracy_df)
    latest_log_time = system_log_df["timestamp"].iloc[0] if not system_log_df.empty else phase3["last_updated"]
    quality_status = data_quality_status()

    status_cols = st.columns(4)
    status_cols[0].metric(tr("history_rows"), f"{len(history_df):,}")
    status_cols[1].metric(tr("total_predictions"), f"{len(model_accuracy_df):,}")
    status_cols[2].metric(tr("best_model"), best_model or tr("not_evaluated"))
    status_cols[3].metric(tr("latest_accuracy"), f"{latest_accuracy:.2%}")

    status_cols_2 = st.columns(4)
    status_cols_2[0].metric(tr("latest_processed_draw"), history_df["date"].iloc[0])
    status_cols_2[1].metric(tr("system_last_refresh"), latest_log_time)
    status_cols_2[2].metric(tr("prediction_records"), f"{len(prediction_history_df):,}")
    status_cols_2[3].metric(tr("data_quality"), quality_status)

    if quality_status == "OK":
        st.success(tr("system_ready"))
    else:
        st.error(f"{tr('status_needs_attention')}: {quality_status}")

    if st.button(tr("refresh_system_status"), type="primary"):
        with st.spinner(tr("refreshing_system")):
            try:
                messages = run_auto_refresh(selected_top_n, selected_rolling_span)
                st.success(f"{tr('refresh_completed')} " + " ".join(messages))
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(tr("refresh_failed"))
                st.code(short_traceback(exc), language="text")

    if not leaderboard_df.empty:
        st.subheader(tr("current_leaderboard"))
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    st.subheader(tr("latest_process_log"))
    st.dataframe(system_log_df, use_container_width=True, hide_index=True)

with tab_backup:
    st.subheader(tr("backup_export"))
    st.info(tr("safety_note"))
    ensure_operational_dirs()
    backup_status = st.session_state.pop("backup_export_status", None)
    if backup_status:
        if backup_status.startswith("Error:"):
            st.error(backup_status)
        else:
            st.success(backup_status)

    storage_mb = folder_size_mb(BACKUP_DIR) + folder_size_mb(EXPORT_DIR)
    backup_cols = st.columns(4)
    backup_cols[0].metric(tr("backup_file_count"), f"{len(backup_df):,}")
    backup_cols[1].metric(tr("latest_export_time"), latest_export_time())
    backup_cols[2].metric(tr("storage_usage_estimate"), f"{storage_mb:.2f} MB")
    backup_cols[3].metric(tr("backup_retention"), f"{tr('latest_count')} {BACKUP_RETENTION_LIMIT}")

    st.subheader(tr("latest_backup_table"))
    if backup_df.empty:
        st.info(tr("no_backup_center_files"))
    else:
        st.dataframe(backup_df.drop(columns=["path"]), use_container_width=True, hide_index=True)

    st.subheader(tr("create_backup"))
    backup_buttons = st.columns(4)
    backup_map = [
        (tr("backup_history"), HISTORY_PATH),
        (tr("backup_accuracy"), MODEL_ACCURACY_PATH),
        (tr("backup_latest_prediction"), LATEST_PREDICTION_PATH),
        (tr("backup_system_log"), SYSTEM_LOG_PATH),
    ]
    for column, (label, path) in zip(backup_buttons, backup_map):
        with column:
            if st.button(label):
                try:
                    backup_path = create_csv_backup(path)
                    append_system_log("manual_backup", "success", f"{path} -> {backup_path}")
                    st.session_state["backup_export_status"] = f"{tr('manual_backup_success')}: {backup_path}"
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.session_state["backup_export_status"] = f"Error: {tr('manual_backup_failed')}: {short_traceback(exc)}"
                    st.rerun()

    st.subheader(tr("export"))
    export_cols = st.columns(4)
    export_requests = [
        (tr("export_predictions_csv"), PREDICTION_HISTORY_PATH, "predictions"),
        (tr("export_accuracy_csv"), MODEL_ACCURACY_PATH, "accuracy"),
        (tr("export_history_csv"), HISTORY_PATH, "history"),
    ]
    for column, (label, path, export_label) in zip(export_cols[:3], export_requests):
        with column:
            if st.button(label):
                try:
                    export_path = export_csv_file(path, export_label)
                    st.session_state["backup_export_status"] = f"{tr('export_created')}: {export_path}"
                    st.session_state["latest_download_path"] = str(export_path)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.session_state["backup_export_status"] = f"Error: {tr('export_failed')}: {short_traceback(exc)}"
                    st.rerun()
    with export_cols[3]:
        if st.button(tr("export_full_system_zip")):
            try:
                export_path = export_full_system_zip()
                st.session_state["backup_export_status"] = f"{tr('full_export_created')}: {export_path}"
                st.session_state["latest_download_path"] = str(export_path)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["backup_export_status"] = f"Error: {tr('full_export_failed')}: {short_traceback(exc)}"
                st.rerun()

    latest_download = st.session_state.get("latest_download_path")
    if latest_download:
        download_path = Path(latest_download)
        if download_path.exists():
            st.download_button(
                tr("download_latest_export"),
                data=download_path.read_bytes(),
                file_name=download_path.name,
                mime="application/zip" if download_path.suffix.lower() == ".zip" else "text/csv",
            )

    st.subheader(tr("restore_safety"))
    st.caption(tr("restore_caption"))
    if backup_df.empty:
        st.info(tr("no_restore_candidates"))
    else:
        backup_names = backup_df["backup_file"].tolist()
        selected_backup = st.selectbox(tr("backup_file"), backup_names)
        selected_source = backup_df.loc[backup_df["backup_file"] == selected_backup, "source_file"].iloc[0]
        target_options = list(BACKUP_TARGETS.keys())
        default_index = target_options.index(selected_source) if selected_source in target_options else 0
        restore_target = st.selectbox(tr("restore_target"), target_options, index=default_index)
        if st.button(tr("validate_backup")):
            try:
                selected_path = BACKUP_DIR / selected_backup
                validate_csv_for_target(selected_path, restore_target)
                st.success(tr("backup_valid"))
            except Exception as exc:
                st.error(f"{tr('backup_validation_failed')}: {short_traceback(exc)}")
        if st.button(tr("restore_selected_backup")):
            try:
                selected_path = BACKUP_DIR / selected_backup
                pre_restore_backup = restore_backup_file(selected_path, restore_target)
                st.session_state["backup_export_status"] = (
                    f"{tr('restore_completed')}: {pre_restore_backup}"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["backup_export_status"] = f"Error: {tr('restore_blocked')}: {short_traceback(exc)}"
                st.rerun()

with tab_model:
    st.subheader(tr("model_comparison"))
    st.info(tr("safety_note"))
    comparison_df = pd.DataFrame(model_comparison["summary"])
    trend_df = pd.DataFrame(rolling_accuracy_trend(model_comparison["details"]))
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    hit_rate_df = comparison_df[["method", "last2_hit_rate", "3digit_hit_rate", "recent_20_round_hit_rate"]].copy()
    hit_rate_df = hit_rate_df.set_index("method")
    st.subheader(tr("hit_rate_comparison"))
    st.plotly_chart(px.bar(hit_rate_df, barmode="group"), use_container_width=True)
    if not trend_df.empty:
        st.subheader(tr("rolling_accuracy_trend"))
        chart_df = trend_df.pivot(index="draw_date", columns="method", values="rolling_hit_rate")
        st.plotly_chart(px.line(chart_df), use_container_width=True)

with tab_backtest:
    st.subheader(tr("backtest"))
    st.info(tr("safety_note"))
    st.write(tr("backtest_note"))
    if st.button(tr("run_backtest"), type="primary"):
        with st.spinner(tr("running_backtest")):
            ok, message, result = run_backtest_export(selected_top_n, selected_rolling_span)
        if ok and result:
            st.success(message)
            st.dataframe(pd.DataFrame(result["summary"]), use_container_width=True, hide_index=True)
        else:
            st.error(tr("backtest_failed"))
            st.code(message, language="text")
    if BACKTEST_RESULT_PATH.exists():
        st.caption(f"{tr('last_result_file')}: {BACKTEST_RESULT_PATH}")
    if BACKTEST_DETAIL_PATH.exists():
        st.caption(f"{tr('detail_csv')}: {BACKTEST_DETAIL_PATH}")

with tab_summary:
    st.subheader(tr("summary"))
    st.info(tr("safety_note"))
    recalc_success = st.session_state.pop("recalculate_success", None)
    if recalc_success:
        st.success(recalc_success)
    if st.button(tr("recalculate_summary"), type="primary"):
        with st.spinner(tr("recalculating_summary")):
            ok, message = recalculate_summary()
        if ok:
            st.session_state["recalculate_success"] = message
            st.rerun()
        else:
            st.error(tr("recalculate_failed"))
            st.code(message, language="text")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab_digits:
    st.subheader(tr("digit_frequency"))
    st.info(tr("safety_note"))
    chart_df = digit_df[["digit", "total"]].copy()
    chart_df["digit"] = chart_df["digit"].astype(str)
    chart_df = chart_df.set_index("digit")
    st.plotly_chart(px.bar(chart_df, y="total"), use_container_width=True)
    st.dataframe(digit_df, use_container_width=True, hide_index=True)

with tab_last2:
    st.subheader(tr("top_last2"))
    st.info(tr("safety_note"))
    st.dataframe(top_last2_df, use_container_width=True, hide_index=True)

with tab_3digit:
    st.subheader(tr("top_3digit"))
    st.info(tr("safety_note"))
    st.dataframe(top_3digit_df, use_container_width=True, hide_index=True)
