from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Lottery History Dashboard",
    page_icon="",
    layout="wide",
)


def read_login_config() -> tuple[str, str]:
    try:
        username = st.secrets["auth"]["username"]
        password = st.secrets["auth"]["password"]
    except Exception:
        st.error("Missing login config")
        st.stop()
    return str(username), str(password)


def require_login() -> None:
    app_username, app_password = read_login_config()
    is_authenticated = (
        st.session_state.get("authenticated") is True
        and st.session_state.get("authenticated_username") == app_username
    )

    if is_authenticated:
        with st.sidebar:
            st.success("Logged in")
            if st.button("Logout"):
                st.session_state.pop("authenticated", None)
                st.session_state.pop("authenticated_username", None)
                st.rerun()
        return

    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == app_username and password == app_password:
            st.session_state["authenticated"] = True
            st.session_state["authenticated_username"] = app_username
            st.rerun()
        st.error("Invalid username or password.")
    st.stop()


require_login()

from predictor import (  # noqa: E402 - imported only after login gate blocks unauthenticated users.
    HYBRID_METHOD,
    METHOD,
    MODEL_ACCURACY_COLUMNS,
    NOTE,
    PREDICTION_HISTORY_COLUMNS,
    SAFETY_NOTE,
    SUPPORTED_METHODS,
    WEIGHTED_METHOD,
    backtest_methods,
    generate_suggestions,
    load_history as predictor_load_history,
    phase3_analysis,
    refresh_model_accuracy,
    read_prediction_history,
    rolling_accuracy_trend,
    run_backtest,
    save_analysis,
    save_model_accuracy_prediction,
)


BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "data" / "lottery_history.csv"
SUMMARY_PATH = BASE_DIR / "output" / "stat_summary.xlsx"
ANALYZER_PATH = BASE_DIR / "analyzer.py"
PREDICTION_HISTORY_PATH = BASE_DIR / "output" / "prediction_history.csv"
MODEL_ACCURACY_PATH = BASE_DIR / "output" / "model_accuracy.csv"
LATEST_PREDICTION_PATH = BASE_DIR / "output" / "latest_prediction.csv"
SYSTEM_LOG_PATH = BASE_DIR / "output" / "system_log.csv"
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
EXPORT_DIR = BASE_DIR / "exports"
SYSTEM_LOG_COLUMNS = ["timestamp", "action", "status", "detail"]
LATEST_PREDICTION_COLUMNS = ["generated_at", "method", "suggested_last2", "suggested_3digit", "source_rows", "note"]
BACKUP_RETENTION_LIMIT = 100
BACKUP_TARGETS = {
    "lottery_history.csv": (HISTORY_PATH, REQUIRED_HISTORY_COLUMNS),
    "prediction_history.csv": (PREDICTION_HISTORY_PATH, PREDICTION_HISTORY_COLUMNS),
    "model_accuracy.csv": (MODEL_ACCURACY_PATH, MODEL_ACCURACY_COLUMNS),
    "latest_prediction.csv": (LATEST_PREDICTION_PATH, LATEST_PREDICTION_COLUMNS),
    "system_log.csv": (SYSTEM_LOG_PATH, SYSTEM_LOG_COLUMNS),
}

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


def load_model_accuracy() -> tuple[pd.DataFrame, dict]:
    result = refresh_model_accuracy(HISTORY_PATH, MODEL_ACCURACY_PATH, BACKUP_DIR)
    rows = result["rows"]
    df = pd.DataFrame(rows, columns=MODEL_ACCURACY_COLUMNS).astype("string").fillna("")
    return df, result


def ensure_operational_dirs() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_backups(keep_latest: int = BACKUP_RETENTION_LIMIT) -> None:
    ensure_operational_dirs()
    backup_files = sorted(
        [path for path in BACKUP_DIR.glob("*.csv") if re.fullmatch(r"\d{8}_\d{6}_.+\.csv", path.name)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for old_file in backup_files[keep_latest:]:
        old_file.unlink()


def center_backup_path(source_path: Path) -> Path:
    ensure_operational_dirs()
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"{stamp}_{source_path.name}"
        if not backup_path.exists():
            return backup_path
        time.sleep(1)


def validate_csv_for_target(path: Path, target_name: str) -> None:
    if target_name == "lottery_history.csv":
        validate_history_csv_file(path)
        return
    target = BACKUP_TARGETS.get(target_name)
    if target is None:
        raise ValueError(f"Unsupported restore target: {target_name}")
    _target_path, columns = target
    ensure_csv_columns(path, columns)


def create_csv_backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.name in BACKUP_TARGETS:
        validate_csv_for_target(path, path.name)
    backup_path = center_backup_path(path)
    shutil.copy2(path, backup_path)
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError(f"Backup failed: {backup_path}")
    cleanup_old_backups()
    return backup_path


def backup_output_file(path: Path, label: str) -> Path | None:
    if path.suffix.lower() == ".csv" and path.name in BACKUP_TARGETS:
        return create_csv_backup(path)
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}__backup__{stamp}__{label}{path.suffix}"
    shutil.copy2(path, backup_path)
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError(f"Backup failed: {backup_path}")
    return backup_path


def ensure_csv_columns(path: Path, columns: list[str]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Invalid CSV header for {path}: expected {columns}, got {reader.fieldnames}")


def append_system_log(action: str, status: str, detail: str) -> None:
    SYSTEM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_columns(SYSTEM_LOG_PATH, SYSTEM_LOG_COLUMNS)
    file_exists = SYSTEM_LOG_PATH.exists()
    with SYSTEM_LOG_PATH.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYSTEM_LOG_COLUMNS, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "status": status,
                "detail": detail,
            }
        )


def list_backup_files() -> pd.DataFrame:
    ensure_operational_dirs()
    rows: list[dict[str, str]] = []
    for path in sorted(BACKUP_DIR.glob("*.csv"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not re.fullmatch(r"\d{8}_\d{6}_.+\.csv", path.name):
            continue
        source_file = re.sub(r"^\d{8}_\d{6}_", "", path.name)
        rows.append(
            {
                "backup_file": path.name,
                "source_file": source_file,
                "size_kb": f"{path.stat().st_size / 1024:.2f}",
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows, columns=["backup_file", "source_file", "size_kb", "modified_at", "path"]).astype("string").fillna("")


def folder_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / (1024 * 1024)


def latest_export_time() -> str:
    ensure_operational_dirs()
    export_files = [path for path in EXPORT_DIR.glob("*") if path.is_file()]
    if not export_files:
        return "No exports yet"
    latest = max(export_files, key=lambda path: path.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime).isoformat(timespec="seconds")


def export_csv_file(source_path: Path, export_label: str) -> Path:
    ensure_operational_dirs()
    if source_path.name in BACKUP_TARGETS:
        validate_csv_for_target(source_path, source_path.name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"{stamp}_{export_label}.csv"
    shutil.copy2(source_path, export_path)
    if not export_path.exists() or export_path.stat().st_size == 0:
        raise RuntimeError(f"Export failed: {export_path}")
    append_system_log("export_csv", "success", f"{source_path} -> {export_path}")
    return export_path


def export_full_system_zip() -> Path:
    ensure_operational_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = EXPORT_DIR / f"{stamp}_lottery_dashboard_full_system.zip"
    files = [
        HISTORY_PATH,
        PREDICTION_HISTORY_PATH,
        MODEL_ACCURACY_PATH,
        LATEST_PREDICTION_PATH,
        SYSTEM_LOG_PATH,
        SUMMARY_PATH,
        BACKTEST_RESULT_PATH,
        BACKTEST_DETAIL_PATH,
        DATA_QUALITY_REPORT_PATH,
    ]
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            if file_path.exists():
                archive.write(file_path, file_path.relative_to(BASE_DIR))
        for backup_path in BACKUP_DIR.glob("*.csv"):
            if backup_path.is_file() and re.fullmatch(r"\d{8}_\d{6}_.+\.csv", backup_path.name):
                archive.write(backup_path, backup_path.relative_to(BASE_DIR))
    if not export_path.exists() or export_path.stat().st_size == 0:
        raise RuntimeError(f"Full system export failed: {export_path}")
    append_system_log("export_full_system_zip", "success", str(export_path))
    return export_path


def restore_backup_file(backup_path: Path, target_name: str) -> Path | None:
    if target_name not in BACKUP_TARGETS:
        raise ValueError(f"Unsupported restore target: {target_name}")
    target_path, _columns = BACKUP_TARGETS[target_name]
    validate_csv_for_target(backup_path, target_name)
    current_backup = create_csv_backup(target_path)
    temp_path = target_path.with_name(f"{target_path.stem}__restore_tmp{target_path.suffix}")
    shutil.copy2(backup_path, temp_path)
    validate_csv_for_target(temp_path, target_name)
    shutil.move(str(temp_path), str(target_path))
    append_system_log("restore_backup", "success", f"{backup_path} -> {target_path}; pre_restore_backup={current_backup}")
    st.cache_data.clear()
    return current_backup


def load_system_log(limit: int | None = None) -> pd.DataFrame:
    if not SYSTEM_LOG_PATH.exists():
        append_system_log("system_log_init", "success", "System log created.")
    df = pd.read_csv(SYSTEM_LOG_PATH, dtype=str, keep_default_na=False)
    df = normalize_text_columns(df, SYSTEM_LOG_COLUMNS)
    df = df.sort_values("timestamp", ascending=False, kind="stable").reset_index(drop=True)
    if limit:
        return df.head(limit)
    return df


def validate_history_dataframe(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_HISTORY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if df[REQUIRED_HISTORY_COLUMNS].isna().any().any():
        raise ValueError("History CSV contains missing values.")
    duplicate_dates = df["date"][df["date"].duplicated()].tolist()
    if duplicate_dates:
        raise ValueError(f"Duplicate draw date(s) after save validation: {', '.join(duplicate_dates)}")
    for column, length in FIELD_LENGTHS.items():
        invalid = df[~df[column].astype(str).str.fullmatch(rf"\d{{{length}}}")]
        if not invalid.empty:
            raise ValueError(f"Invalid {column} value found during save validation.")


def validate_history_csv_file(path: Path) -> None:
    candidate = pd.read_csv(path, dtype=str, keep_default_na=False)
    candidate = normalize_text_columns(candidate[REQUIRED_HISTORY_COLUMNS], REQUIRED_HISTORY_COLUMNS)
    candidate["date"] = candidate["date"].apply(normalize_date)
    for column, length in FIELD_LENGTHS.items():
        candidate[column] = candidate[column].apply(lambda value, col=column, size=length: normalize_number(value, size, col))
    validate_history_dataframe(candidate)


def restore_history_from_backup(backup_path: Path) -> None:
    if not backup_path.exists():
        raise RuntimeError(f"Rollback backup is missing: {backup_path}")
    shutil.copy2(backup_path, HISTORY_PATH)
    validate_history_csv_file(HISTORY_PATH)


def generate_latest_prediction_csv(top_n: int) -> Path | None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, str]] = []
    for method in SUPPORTED_METHODS:
        suggestions = generate_suggestions(HISTORY_PATH, method=method, top_n=top_n)
        rows.append(
            {
                "generated_at": generated_at,
                "method": method,
                "suggested_last2": ",".join(item["number"] for item in suggestions["suggested_last2"]),
                "suggested_3digit": ",".join(item["number"] for item in suggestions["suggested_3digit"]),
                "source_rows": str(suggestions["source_rows"]),
                "note": SAFETY_NOTE,
            }
        )

    backup_path = backup_output_file(LATEST_PREDICTION_PATH, "latest_prediction")
    LATEST_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = LATEST_PREDICTION_PATH.with_name(f"{LATEST_PREDICTION_PATH.stem}__tmp{LATEST_PREDICTION_PATH.suffix}")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATEST_PREDICTION_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    ensure_csv_columns(temp_path, LATEST_PREDICTION_COLUMNS)
    shutil.move(str(temp_path), str(LATEST_PREDICTION_PATH))
    return backup_path


def data_quality_status() -> str:
    if not DATA_QUALITY_REPORT_PATH.exists():
        return "Missing data quality report"
    try:
        summary = pd.read_excel(DATA_QUALITY_REPORT_PATH, sheet_name="Summary", dtype=str, keep_default_na=False, engine="openpyxl")
        metric_map = {str(row.get("metric", "")).strip(): str(row.get("value", "")).strip() for _index, row in summary.iterrows()}
        critical_count = int(float(metric_map.get("critical_count", "0") or 0))
        warning_count = int(float(metric_map.get("warning_count", "0") or 0))
        if critical_count:
            return f"Critical issues: {critical_count}"
        if warning_count:
            return f"Warnings: {warning_count}"
        return "OK"
    except Exception as exc:
        return f"Unreadable report: {short_traceback(exc)}"


def accuracy_leaderboard(df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    work = df.copy()
    for column in ["hit_last2", "hit_3digit", "score"]:
        work[f"{column}_num"] = pd.to_numeric(work[column], errors="coerce")
    evaluated = work[work["score_num"].notna()].copy()
    if evaluated.empty:
        return pd.DataFrame(), 0.0, ""
    total_hit_units = evaluated["hit_last2_num"].sum() + evaluated["hit_3digit_num"].sum()
    latest_accuracy = total_hit_units / (len(evaluated) * 2)
    leaderboard = (
        evaluated.groupby("method", dropna=False)
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
    return leaderboard, float(latest_accuracy), str(leaderboard.iloc[0]["method"])


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
    backup_path = create_csv_backup(path)
    if backup_path is None:
        raise RuntimeError(f"Backup source does not exist: {path}")
    return backup_path


def save_new_lottery_row(row: dict[str, str], history: pd.DataFrame) -> Path:
    backup_path = backup_history_csv(HISTORY_PATH)
    new_df = pd.concat([pd.DataFrame([row], columns=REQUIRED_HISTORY_COLUMNS), history], ignore_index=True)
    new_df = normalize_text_columns(new_df[REQUIRED_HISTORY_COLUMNS], REQUIRED_HISTORY_COLUMNS)
    new_df["date"] = new_df["date"].apply(normalize_date)
    new_df = new_df.drop_duplicates(subset=["date"], keep="first")
    new_df = new_df.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)
    validate_history_dataframe(new_df)
    temp_path = HISTORY_PATH.with_name(f"{HISTORY_PATH.stem}__streamlit_tmp{HISTORY_PATH.suffix}")
    new_df.to_csv(temp_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    validate_history_csv_file(temp_path)
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
    messages.append("Model accuracy and leaderboard refreshed.")

    append_system_log("refresh_prediction_history", "started", "Validating prediction history file.")
    prediction_rows = read_prediction_history(PREDICTION_HISTORY_PATH)
    append_system_log("refresh_prediction_history", "success", f"records={len(prediction_rows)}")
    messages.append("Prediction history refreshed.")

    append_system_log("generate_latest_prediction", "started", "Generating latest statistical suggestions.")
    latest_backup = generate_latest_prediction_csv(top_n)
    latest_detail = f"latest_prediction={LATEST_PREDICTION_PATH}"
    if latest_backup:
        latest_detail += f"; backup={latest_backup}"
    append_system_log("generate_latest_prediction", "success", latest_detail)
    messages.append("Latest prediction generated.")

    append_system_log("refresh_backtest", "started", "Running rolling backtest and model comparison outputs.")
    backtest_ok, backtest_message, _result = run_backtest_export(top_n, rolling_span)
    if not backtest_ok:
        append_system_log("refresh_backtest", "failed", backtest_message)
        raise RuntimeError(backtest_message)
    append_system_log("refresh_backtest", "success", backtest_message)
    messages.append("Backtest and model comparison refreshed.")

    st.cache_data.clear()
    return messages


def process_new_result(row: dict[str, str], history: pd.DataFrame, top_n: int, rolling_span: int) -> list[str]:
    backup_path: Path | None = None
    try:
        append_system_log("add_result", "started", f"draw_date={row['date']}")
        backup_path = save_new_lottery_row(row, history)
        append_system_log("add_result", "success", f"draw_date={row['date']}; backup={backup_path}")
        messages = [f"Saved result. Backup created: {backup_path}"]
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
    model_accuracy_df, model_accuracy_result = load_model_accuracy()
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
metric_cols[0].metric("History Rows", f"{len(history_df):,}")
metric_cols[1].metric("Latest Draw", history_df["date"].iloc[0])
metric_cols[2].metric("Current Method", selected_method)
metric_cols[3].metric("Last Updated", phase3["last_updated"])

tab_history, tab_add, tab_next, tab_prediction_history, tab_accuracy, tab_status, tab_backup, tab_model, tab_backtest, tab_summary, tab_digits, tab_last2, tab_3digit = st.tabs(
    [
        "Lottery History",
        "Add Result",
        "Next Draw Analysis",
        "Prediction History",
        "Model Accuracy",
        "System Status",
        "Backup & Export",
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
                process_messages = process_new_result(new_row, history_df, selected_top_n, selected_rolling_span)
                st.session_state["add_result_success"] = "Auto processing completed. " + " ".join(process_messages)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Auto processing failed. Changes were rolled back when possible: {exc}")

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
                st.session_state["save_analysis_success"] = f"Analysis saved. Backup created: {', '.join(backup_notes)}"
            else:
                st.session_state["save_analysis_success"] = "Analysis saved. Tracking files created."
            st.rerun()
        except Exception as exc:
            st.error(f"Save Analysis failed: {exc}")

    st.caption(NOTE)

with tab_prediction_history:
    st.subheader("Prediction History")
    st.info(NOTE)
    st.dataframe(prediction_history_df, use_container_width=True, hide_index=True)

with tab_accuracy:
    st.subheader("Model Accuracy")
    st.info(NOTE)

    total_predictions = len(model_accuracy_df)
    accuracy_work = model_accuracy_df.copy()
    for column in ["hit_last2", "hit_3digit", "score"]:
        accuracy_work[f"{column}_num"] = pd.to_numeric(accuracy_work[column], errors="coerce")
    evaluated_df = accuracy_work[accuracy_work["score_num"].notna()].copy()
    evaluated_count = len(evaluated_df)
    total_hit_units = int(evaluated_df["hit_last2_num"].sum() + evaluated_df["hit_3digit_num"].sum()) if evaluated_count else 0
    overall_hit_rate = total_hit_units / (evaluated_count * 2) if evaluated_count else 0

    accuracy_cols = st.columns(4)
    accuracy_cols[0].metric("Total Predictions", f"{total_predictions:,}")
    accuracy_cols[1].metric("Evaluated", f"{evaluated_count:,}")
    accuracy_cols[2].metric("Hit Rate %", f"{overall_hit_rate:.2%}")
    accuracy_cols[3].metric("Tracking File", MODEL_ACCURACY_PATH.name)

    if model_accuracy_result.get("updated"):
        st.success("Model accuracy refreshed against available actual results.")
    if model_accuracy_result.get("backup_path"):
        st.caption(f"Backup created: {model_accuracy_result['backup_path']}")

    st.subheader("Accuracy Table")
    st.dataframe(model_accuracy_df, use_container_width=True, hide_index=True)

    if evaluated_df.empty:
        st.info("No evaluated prediction records yet. Save an analysis for a draw date, then add the actual result when available.")
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
        st.success(f"Best current model by historical tracking: {best_method}")

        def highlight_best(row: pd.Series) -> list[str]:
            return ["background-color: #d9f7be" if row["rank"] == 1 else "" for _value in row]

        st.subheader("Leaderboard Model Ranking")
        st.dataframe(leaderboard.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

        rate_chart_df = leaderboard[["method", "last2_hit_rate", "3digit_hit_rate", "overall_hit_rate"]].melt(
            id_vars="method",
            var_name="metric",
            value_name="hit_rate",
        )
        st.subheader("Hit Rate Comparison")
        st.plotly_chart(px.bar(rate_chart_df, x="method", y="hit_rate", color="metric", barmode="group"), use_container_width=True)

        trend_df = evaluated_df.sort_values(["method", "draw_date", "created_at"], kind="stable").copy()
        trend_df["accuracy_point"] = trend_df["score_num"] / 2
        trend_df["accuracy_trend"] = (
            trend_df.groupby("method")["accuracy_point"].expanding().mean().reset_index(level=0, drop=True)
        )
        st.subheader("Accuracy Trend")
        st.plotly_chart(
            px.line(trend_df, x="draw_date", y="accuracy_trend", color="method", markers=True),
            use_container_width=True,
        )

with tab_status:
    st.subheader("System Status")
    st.info(NOTE)
    leaderboard_df, latest_accuracy, best_model = accuracy_leaderboard(model_accuracy_df)
    latest_log_time = system_log_df["timestamp"].iloc[0] if not system_log_df.empty else phase3["last_updated"]
    quality_status = data_quality_status()

    status_cols = st.columns(4)
    status_cols[0].metric("Total Draws", f"{len(history_df):,}")
    status_cols[1].metric("Total Predictions", f"{len(model_accuracy_df):,}")
    status_cols[2].metric("Best Model", best_model or "Not evaluated")
    status_cols[3].metric("Latest Accuracy", f"{latest_accuracy:.2%}")

    status_cols_2 = st.columns(4)
    status_cols_2[0].metric("Latest Processed Draw", history_df["date"].iloc[0])
    status_cols_2[1].metric("System Last Refresh", latest_log_time)
    status_cols_2[2].metric("Prediction Records", f"{len(prediction_history_df):,}")
    status_cols_2[3].metric("Data Quality", quality_status)

    if quality_status == "OK":
        st.success("System status is ready. Historical statistical analysis only.")
    else:
        st.error(f"System status needs attention: {quality_status}")

    if st.button("Refresh System Status", type="primary"):
        with st.spinner("Refreshing statistics, accuracy, leaderboard, and latest prediction..."):
            try:
                messages = run_auto_refresh(selected_top_n, selected_rolling_span)
                st.success("Refresh completed. " + " ".join(messages))
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error("Refresh failed.")
                st.code(short_traceback(exc), language="text")

    if not leaderboard_df.empty:
        st.subheader("Current Leaderboard")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    st.subheader("Latest Process Log")
    st.dataframe(system_log_df, use_container_width=True, hide_index=True)

with tab_backup:
    st.subheader("Backup & Export")
    st.info(NOTE)
    ensure_operational_dirs()
    backup_status = st.session_state.pop("backup_export_status", None)
    if backup_status:
        if backup_status.startswith("Error:"):
            st.error(backup_status)
        else:
            st.success(backup_status)

    storage_mb = folder_size_mb(BACKUP_DIR) + folder_size_mb(EXPORT_DIR)
    backup_cols = st.columns(4)
    backup_cols[0].metric("Backup File Count", f"{len(backup_df):,}")
    backup_cols[1].metric("Latest Export Time", latest_export_time())
    backup_cols[2].metric("Storage Usage Estimate", f"{storage_mb:.2f} MB")
    backup_cols[3].metric("Backup Retention", f"Latest {BACKUP_RETENTION_LIMIT}")

    st.subheader("Latest Backup Table")
    if backup_df.empty:
        st.info("No Backup Center CSV backups yet.")
    else:
        st.dataframe(backup_df.drop(columns=["path"]), use_container_width=True, hide_index=True)

    st.subheader("Create Backup")
    backup_buttons = st.columns(4)
    backup_map = [
        ("Backup History", HISTORY_PATH),
        ("Backup Accuracy", MODEL_ACCURACY_PATH),
        ("Backup Latest Prediction", LATEST_PREDICTION_PATH),
        ("Backup System Log", SYSTEM_LOG_PATH),
    ]
    for column, (label, path) in zip(backup_buttons, backup_map):
        with column:
            if st.button(label):
                try:
                    backup_path = create_csv_backup(path)
                    append_system_log("manual_backup", "success", f"{path} -> {backup_path}")
                    st.session_state["backup_export_status"] = f"Backup created: {backup_path}"
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.session_state["backup_export_status"] = f"Error: Backup failed: {short_traceback(exc)}"
                    st.rerun()

    st.subheader("Export")
    export_cols = st.columns(4)
    export_requests = [
        ("Export Predictions CSV", PREDICTION_HISTORY_PATH, "predictions"),
        ("Export Accuracy CSV", MODEL_ACCURACY_PATH, "accuracy"),
        ("Export History CSV", HISTORY_PATH, "history"),
    ]
    for column, (label, path, export_label) in zip(export_cols[:3], export_requests):
        with column:
            if st.button(label):
                try:
                    export_path = export_csv_file(path, export_label)
                    st.session_state["backup_export_status"] = f"Export created: {export_path}"
                    st.session_state["latest_download_path"] = str(export_path)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.session_state["backup_export_status"] = f"Error: Export failed: {short_traceback(exc)}"
                    st.rerun()
    with export_cols[3]:
        if st.button("Export Full System ZIP"):
            try:
                export_path = export_full_system_zip()
                st.session_state["backup_export_status"] = f"Full system export created: {export_path}"
                st.session_state["latest_download_path"] = str(export_path)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["backup_export_status"] = f"Error: Full export failed: {short_traceback(exc)}"
                st.rerun()

    latest_download = st.session_state.get("latest_download_path")
    if latest_download:
        download_path = Path(latest_download)
        if download_path.exists():
            st.download_button(
                "Download Latest Export",
                data=download_path.read_bytes(),
                file_name=download_path.name,
                mime="application/zip" if download_path.suffix.lower() == ".zip" else "text/csv",
            )

    st.subheader("Restore Safety")
    st.caption("Restore validates CSV structure before replacing any live file.")
    if backup_df.empty:
        st.info("No restore candidates available.")
    else:
        backup_names = backup_df["backup_file"].tolist()
        selected_backup = st.selectbox("Backup file", backup_names)
        selected_source = backup_df.loc[backup_df["backup_file"] == selected_backup, "source_file"].iloc[0]
        target_options = list(BACKUP_TARGETS.keys())
        default_index = target_options.index(selected_source) if selected_source in target_options else 0
        restore_target = st.selectbox("Restore target", target_options, index=default_index)
        if st.button("Validate Backup"):
            try:
                selected_path = BACKUP_DIR / selected_backup
                validate_csv_for_target(selected_path, restore_target)
                st.success("Backup structure is valid for the selected restore target.")
            except Exception as exc:
                st.error(f"Backup validation failed: {short_traceback(exc)}")
        if st.button("Restore Selected Backup"):
            try:
                selected_path = BACKUP_DIR / selected_backup
                pre_restore_backup = restore_backup_file(selected_path, restore_target)
                st.session_state["backup_export_status"] = (
                    f"Restore completed. Current file was backed up first: {pre_restore_backup}"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["backup_export_status"] = f"Error: Restore blocked: {short_traceback(exc)}"
                st.rerun()

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
