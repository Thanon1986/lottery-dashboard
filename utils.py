"""Utility layer for file safety, CSV IO, exports, logging, and insights."""

from __future__ import annotations

import csv
import re
import shutil
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from constants import (
    BACKTEST_DETAIL_PATH,
    BACKTEST_RESULT_PATH,
    BACKUP_DIR,
    BACKUP_RETENTION_LIMIT,
    BACKUP_TARGETS,
    BASE_DIR,
    DATA_QUALITY_REPORT_PATH,
    EXPORT_DIR,
    FIELD_LENGTHS,
    HISTORY_PATH,
    INSIGHT_HISTORY_COLUMNS,
    INSIGHT_HISTORY_PATH,
    INSIGHT_MOVEMENT_BUCKET_SIZE,
    INSIGHT_PRIOR_WINDOW,
    INSIGHT_RECENT_WINDOW,
    LATEST_PREDICTION_COLUMNS,
    LATEST_PREDICTION_PATH,
    MODEL_ACCURACY_PATH,
    NUMBER_COLUMNS,
    PREDICTION_HISTORY_PATH,
    REQUIRED_HISTORY_COLUMNS,
    SIGNAL_HIGH_THRESHOLD,
    SIGNAL_MEDIUM_THRESHOLD,
    SUMMARY_PATH,
    SYSTEM_LOG_COLUMNS,
    SYSTEM_LOG_PATH,
)
from predictor import SAFETY_NOTE, SUPPORTED_METHODS, generate_suggestions


def short_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].astype("string").fillna("").str.strip()
    return result


def numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def safe_read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot read CSV {path}: {short_traceback(exc)}") from exc
    if columns:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    return df


def safe_write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}__tmp{path.suffix}")
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])
        ensure_csv_columns(temp_path, columns)
        shutil.move(str(temp_path), str(path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def normalize_date(value: str) -> str:
    raw_value = str(value).strip()
    for date_format in ["%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(raw_value, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format {raw_value!r}; expected YYYY-MM-DD or DD/MM/YYYY")


def normalize_number(value: str, length: int, column: str) -> str:
    raw_value = str(value).strip()
    if not raw_value.isdigit():
        raise ValueError(f"Invalid {column} value {raw_value!r}; digits only")
    if len(raw_value) > length:
        raise ValueError(f"Invalid {column} value {raw_value!r}; max {length} digits")
    return raw_value.zfill(length)


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


def ensure_csv_columns(path: Path, columns: list[str]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Invalid CSV header for {path}: expected {columns}, got {reader.fieldnames}")


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
    candidate = safe_read_csv(path, REQUIRED_HISTORY_COLUMNS)
    candidate = normalize_text_columns(candidate[REQUIRED_HISTORY_COLUMNS], REQUIRED_HISTORY_COLUMNS)
    candidate["date"] = candidate["date"].apply(normalize_date)
    for column, length in FIELD_LENGTHS.items():
        candidate[column] = candidate[column].apply(lambda value, col=column, size=length: normalize_number(value, size, col))
    validate_history_dataframe(candidate)


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
        INSIGHT_HISTORY_PATH,
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
    return current_backup


def ensure_insight_history() -> None:
    INSIGHT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INSIGHT_HISTORY_PATH.exists():
        ensure_csv_columns(INSIGHT_HISTORY_PATH, INSIGHT_HISTORY_COLUMNS)
        return
    safe_write_csv(INSIGHT_HISTORY_PATH, [], INSIGHT_HISTORY_COLUMNS)


def read_insight_history() -> pd.DataFrame:
    ensure_insight_history()
    df = safe_read_csv(INSIGHT_HISTORY_PATH, INSIGHT_HISTORY_COLUMNS)
    return normalize_text_columns(df, INSIGHT_HISTORY_COLUMNS)


def load_system_log(limit: int | None = None) -> pd.DataFrame:
    if not SYSTEM_LOG_PATH.exists():
        append_system_log("system_log_init", "success", "System log created.")
    df = safe_read_csv(SYSTEM_LOG_PATH, SYSTEM_LOG_COLUMNS)
    df = normalize_text_columns(df, SYSTEM_LOG_COLUMNS)
    df = df.sort_values("timestamp", ascending=False, kind="stable").reset_index(drop=True)
    if limit:
        return df.head(limit)
    return df


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
    safe_write_csv(LATEST_PREDICTION_PATH, rows, LATEST_PREDICTION_COLUMNS)
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


def bounded_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def signal_from_confidence(confidence: int) -> str:
    if confidence >= SIGNAL_HIGH_THRESHOLD:
        return "HIGH"
    if confidence >= SIGNAL_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def insight_record(category: str, title: str, confidence: int, explanation: str, warnings: list[str], generated_at: str, latest_draw: str) -> dict[str, str]:
    confidence = bounded_score(confidence)
    return {
        "generated_at": generated_at,
        "latest_draw": latest_draw,
        "category": category,
        "title": title,
        "signal": signal_from_confidence(confidence),
        "confidence": str(confidence),
        "score": str(confidence),
        "explanation": explanation,
        "warnings": "; ".join(warnings),
        "note": SAFETY_NOTE,
    }


def all_lottery_digits(rows: pd.DataFrame) -> list[str]:
    digits: list[str] = []
    for column in NUMBER_COLUMNS:
        for value in rows[column].astype(str):
            digits.extend([digit for digit in value if digit.isdigit()])
    return digits


def three_digit_values(rows: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for column in ["front3_1", "front3_2", "back3_1", "back3_2"]:
        values.extend(rows[column].astype(str).tolist())
    return values


def top_trending(counter_recent: Counter[str], counter_prior: Counter[str], recent_size: int, prior_size: int, limit: int = 3) -> list[tuple[str, float, int]]:
    scores: list[tuple[str, float, int]] = []
    candidates = set(counter_recent) | set(counter_prior)
    for value in candidates:
        recent_rate = counter_recent[value] / max(recent_size, 1)
        prior_rate = counter_prior[value] / max(prior_size, 1)
        scores.append((value, recent_rate - prior_rate, counter_recent[value]))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]


def build_ai_insights(history: pd.DataFrame) -> dict[str, object]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    latest_draw = str(history["date"].iloc[0]) if not history.empty else ""
    row_count = len(history)
    recent_window = min(INSIGHT_RECENT_WINDOW, row_count)
    prior_window = min(INSIGHT_PRIOR_WINDOW, max(row_count - recent_window, 0))
    recent = history.head(recent_window)
    prior = history.iloc[recent_window : recent_window + prior_window]
    warnings: list[str] = []
    if row_count < 50:
        warnings.append("low data confidence")
    if row_count < 20:
        warnings.append("insufficient history")
    if prior.empty:
        warnings.append("unstable pattern")

    insights: list[dict[str, str]] = []
    all_digits = all_lottery_digits(history)
    recent_digits = all_lottery_digits(recent)
    prior_digits = all_lottery_digits(prior) if not prior.empty else all_digits
    digit_counts = Counter(all_digits)
    recent_digit_counts = Counter(recent_digits)
    prior_digit_counts = Counter(prior_digits)

    hot_digit, hot_count = sorted(digit_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    cold_digit, cold_count = sorted(digit_counts.items(), key=lambda item: (item[1], item[0]))[0]
    avg_digit_count = sum(digit_counts.values()) / max(len(digit_counts), 1)
    hot_confidence = 45 + ((hot_count - avg_digit_count) / max(avg_digit_count, 1) * 35) + min(row_count, 100) * 0.15
    cold_confidence = 45 + ((avg_digit_count - cold_count) / max(avg_digit_count, 1) * 35) + min(row_count, 100) * 0.15
    insights.append(insight_record("hot_digits", f"Hot digit: {hot_digit}", hot_confidence, f"Digit {hot_digit} appears {hot_count} times across historical prize fields.", warnings, generated_at, latest_draw))
    insights.append(insight_record("cold_digits", f"Cold digit: {cold_digit}", cold_confidence, f"Digit {cold_digit} has the lowest historical count at {cold_count} appearances.", warnings, generated_at, latest_draw))

    recent_last2 = Counter(recent["last2"].astype(str))
    prior_last2 = Counter(prior["last2"].astype(str)) if not prior.empty else Counter(history["last2"].astype(str))
    trending_last2 = top_trending(recent_last2, prior_last2, recent_window, max(len(prior), 1), limit=3)
    last2_text = ", ".join(value for value, _score, _count in trending_last2)
    last2_confidence = 50 + min(max(trending_last2[0][1], 0) * 180, 35) + min(row_count, 80) * 0.1 if trending_last2 else 35
    insights.append(insight_record("trending_last2", f"Trending last2: {last2_text or 'none'}", last2_confidence, f"Recent last2 values with the strongest frequency lift are {last2_text or 'not available'}.", warnings, generated_at, latest_draw))

    recent_3digit = Counter(three_digit_values(recent))
    prior_3digit = Counter(three_digit_values(prior)) if not prior.empty else Counter(three_digit_values(history))
    trending_3digit = top_trending(recent_3digit, prior_3digit, recent_window * 4, max(len(prior) * 4, 1), limit=3)
    three_text = ", ".join(value for value, _score, _count in trending_3digit)
    three_confidence = 48 + min(max(trending_3digit[0][1], 0) * 260, 35) + min(row_count, 80) * 0.1 if trending_3digit else 35
    insights.append(insight_record("trending_3digit", f"Trending 3digit: {three_text or 'none'}", three_confidence, f"Recent 3digit values with the strongest frequency lift are {three_text or 'not available'}.", warnings, generated_at, latest_draw))

    digit_lifts = top_trending(recent_digit_counts, prior_digit_counts, max(len(recent_digits), 1), max(len(prior_digits), 1), limit=1)
    spike_digit, spike_lift, _count = digit_lifts[0]
    spike_confidence = 45 + min(max(spike_lift, 0) * 320, 40) + min(row_count, 100) * 0.1
    insights.append(insight_record("frequency_spike", f"Frequency spike: digit {spike_digit}", spike_confidence, f"Digit {spike_digit} increased by {spike_lift:.2%} in the recent window versus the comparison window.", warnings, generated_at, latest_draw))

    repeat_counter = Counter(history["last2"].astype(str).head(40))
    repeated = sorted([(value, count) for value, count in repeat_counter.items() if count >= 2], key=lambda item: (-item[1], item[0]))
    repeat_title = f"Repeating last2: {repeated[0][0]}" if repeated else "Repeating pattern: weak"
    repeat_explanation = f"Last2 {repeated[0][0]} appears {repeated[0][1]} times in the recent 40 draws." if repeated else "No strong repeated last2 pattern is visible in the recent 40 draws."
    repeat_confidence = 55 + min((repeated[0][1] - 1) * 10, 35) if repeated else 35
    insights.append(insight_record("repeating_pattern", repeat_title, repeat_confidence, repeat_explanation, warnings, generated_at, latest_draw))

    missing_digits = sorted(set(str(digit) for digit in range(10)) - set(recent_digit_counts.keys()))
    missing_confidence = 80 if missing_digits else 42
    missing_text = ", ".join(missing_digits) if missing_digits else "none"
    missing_warnings = warnings + (["missing digit anomaly"] if missing_digits else [])
    insights.append(insight_record("missing_digit_anomaly", f"Missing recent digits: {missing_text}", missing_confidence, f"Digits absent from the recent {recent_window} draws: {missing_text}.", missing_warnings, generated_at, latest_draw))

    shift_amount = sum(abs((recent_digit_counts[str(d)] / max(len(recent_digits), 1)) - (prior_digit_counts[str(d)] / max(len(prior_digits), 1))) for d in range(10))
    shift_confidence = 45 + min(shift_amount * 130, 45) + min(row_count, 100) * 0.05
    shift_warnings = warnings + (["unstable pattern"] if shift_amount > 0.28 else [])
    insights.append(insight_record("trend_shift", "Digit distribution trend shift", shift_confidence, f"Recent digit distribution differs from the comparison window by {shift_amount:.2f} total movement.", shift_warnings, generated_at, latest_draw))

    return {
        "generated_at": generated_at,
        "latest_draw": latest_draw,
        "insights": insights,
        "warnings": sorted(set(warnings)),
        "heatmap": digit_heatmap(history),
        "trend": trend_chart_data(history),
        "movement": digit_movement_data(history),
    }


def digit_heatmap(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in NUMBER_COLUMNS:
        values = "".join(history[column].astype(str).tolist())
        counter = Counter(values)
        for digit in [str(value) for value in range(10)]:
            rows.append({"field": column, "digit": digit, "count": counter[digit]})
    return pd.DataFrame(rows)


def trend_chart_data(history: pd.DataFrame) -> pd.DataFrame:
    chronological = history.sort_values("date", ascending=True, kind="stable").reset_index(drop=True)
    top_values = [value for value, _count in Counter(history.head(40)["last2"].astype(str)).most_common(5)]
    rows: list[dict[str, object]] = []
    running = {value: 0 for value in top_values}
    for _index, row in chronological.iterrows():
        last2 = str(row["last2"])
        if last2 in running:
            running[last2] += 1
        for value in top_values:
            rows.append({"date": row["date"], "last2": value, "cumulative_count": running[value]})
    return pd.DataFrame(rows)


def digit_movement_data(history: pd.DataFrame, bucket_size: int = INSIGHT_MOVEMENT_BUCKET_SIZE) -> pd.DataFrame:
    chronological = history.sort_values("date", ascending=True, kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for start in range(0, len(chronological), bucket_size):
        bucket = chronological.iloc[start : start + bucket_size]
        if bucket.empty:
            continue
        label = f"{bucket['date'].iloc[0]} to {bucket['date'].iloc[-1]}"
        counter = Counter(all_lottery_digits(bucket))
        total = sum(counter.values())
        for digit in [str(value) for value in range(10)]:
            rows.append({"period": label, "digit": digit, "share": counter[digit] / total if total else 0})
    return pd.DataFrame(rows)


def save_insight_history_if_new(insight_result: dict[str, object]) -> dict[str, object]:
    ensure_insight_history()
    existing = read_insight_history()
    insights = insight_result["insights"]
    latest_draw = str(insight_result["latest_draw"])
    new_rows = [{column: str(insight.get(column, "")) for column in INSIGHT_HISTORY_COLUMNS} for insight in insights]  # type: ignore[union-attr]
    existing_keys = set(zip(existing["latest_draw"], existing["category"], existing["title"])) if not existing.empty else set()
    rows_to_add = [row for row in new_rows if (row["latest_draw"], row["category"], row["title"]) not in existing_keys]
    if not rows_to_add:
        return {"updated": False, "added": 0, "backup_path": "", "latest_draw": latest_draw}

    backup_path = create_csv_backup(INSIGHT_HISTORY_PATH)
    output_rows = existing.to_dict("records") + rows_to_add
    safe_write_csv(INSIGHT_HISTORY_PATH, output_rows, INSIGHT_HISTORY_COLUMNS)
    append_system_log("ai_insights", "success", f"added={len(rows_to_add)}; latest_draw={latest_draw}")
    return {"updated": True, "added": len(rows_to_add), "backup_path": str(backup_path) if backup_path else "", "latest_draw": latest_draw}


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
