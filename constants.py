"""Shared constants for the Lottery Statistics Dashboard."""

from __future__ import annotations

from pathlib import Path

from predictor import MODEL_ACCURACY_COLUMNS, PREDICTION_HISTORY_COLUMNS


BASE_DIR = Path(__file__).resolve().parent

HISTORY_PATH = BASE_DIR / "data" / "lottery_history.csv"
LOTTERY_HISTORY_PATH = HISTORY_PATH
SUMMARY_PATH = BASE_DIR / "output" / "stat_summary.xlsx"
ANALYZER_PATH = BASE_DIR / "analyzer.py"
PREDICTION_HISTORY_PATH = BASE_DIR / "output" / "prediction_history.csv"
MODEL_ACCURACY_PATH = BASE_DIR / "output" / "model_accuracy.csv"
LATEST_PREDICTION_PATH = BASE_DIR / "output" / "latest_prediction.csv"
SYSTEM_LOG_PATH = BASE_DIR / "output" / "system_log.csv"
INSIGHT_HISTORY_PATH = BASE_DIR / "output" / "insight_history.csv"
BACKTEST_RESULT_PATH = BASE_DIR / "output" / "backtest_result.xlsx"
BACKTEST_DETAIL_PATH = BASE_DIR / "output" / "backtest_detail.csv"
DATA_QUALITY_REPORT_PATH = BASE_DIR / "output" / "data_quality_report.xlsx"

BACKUP_DIR = BASE_DIR / "backup"
EXPORT_DIR = BASE_DIR / "exports"

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

SYSTEM_LOG_COLUMNS = ["timestamp", "action", "status", "detail"]
LATEST_PREDICTION_COLUMNS = ["generated_at", "method", "suggested_last2", "suggested_3digit", "source_rows", "note"]
INSIGHT_HISTORY_COLUMNS = [
    "generated_at",
    "latest_draw",
    "category",
    "title",
    "signal",
    "confidence",
    "score",
    "explanation",
    "warnings",
    "note",
]

BACKUP_RETENTION_LIMIT = 100
SIGNAL_HIGH_THRESHOLD = 75
SIGNAL_MEDIUM_THRESHOLD = 50
INSIGHT_RECENT_WINDOW = 20
INSIGHT_PRIOR_WINDOW = 60
INSIGHT_MOVEMENT_BUCKET_SIZE = 10

BACKUP_TARGETS = {
    "lottery_history.csv": (HISTORY_PATH, REQUIRED_HISTORY_COLUMNS),
    "prediction_history.csv": (PREDICTION_HISTORY_PATH, PREDICTION_HISTORY_COLUMNS),
    "model_accuracy.csv": (MODEL_ACCURACY_PATH, MODEL_ACCURACY_COLUMNS),
    "latest_prediction.csv": (LATEST_PREDICTION_PATH, LATEST_PREDICTION_COLUMNS),
    "system_log.csv": (SYSTEM_LOG_PATH, SYSTEM_LOG_COLUMNS),
    "insight_history.csv": (INSIGHT_HISTORY_PATH, INSIGHT_HISTORY_COLUMNS),
}
