#!/usr/bin/env python3
"""
Statistical suggestion utilities for the Lottery Dashboard.

All calculations use historical data only. The module is deterministic, reads
the source CSV as text, and never modifies lottery_history.csv.
"""

from __future__ import annotations

import csv
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


METHOD = "frequency_score_v1"
WEIGHTED_METHOD = "weighted_recent_v1"
HYBRID_METHOD = "hybrid_score_v1"
SUPPORTED_METHODS = [METHOD, WEIGHTED_METHOD, HYBRID_METHOD]
SAFETY_NOTE = "Historical statistical analysis only. This system does not predict or guarantee lottery outcomes."
NOTE = SAFETY_NOTE

REQUIRED_COLUMNS = ["date", "first_prize", "last2", "front3_1", "front3_2", "back3_1", "back3_2"]
FIELD_LENGTHS = {
    "first_prize": 6,
    "last2": 2,
    "front3_1": 3,
    "front3_2": 3,
    "back3_1": 3,
    "back3_2": 3,
}
THREE_DIGIT_COLUMNS = ["front3_1", "front3_2", "back3_1", "back3_2"]
PREDICTION_HISTORY_COLUMNS = [
    "created_at",
    "target_draw",
    "method",
    "suggested_last2",
    "suggested_3digit",
    "source_rows",
    "note",
]
BACKTEST_DETAIL_COLUMNS = [
    "draw_date",
    "method",
    "suggested_last2",
    "actual_last2",
    "hit_last2",
    "suggested_3digit",
    "actual_3digit",
    "hit_3digit",
    "rank_hit",
]


def normalize_date(value: str) -> str:
    """Normalize YYYY-MM-DD or DD/MM/YYYY to YYYY-MM-DD."""
    raw_value = str(value).strip()
    for date_format in ["%Y-%m-%d", "%d/%m/%Y"]:
        try:
            return datetime.strptime(raw_value, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Invalid date format {raw_value!r}; expected YYYY-MM-DD or DD/MM/YYYY")


def normalize_number(value: str, length: int, column: str, row_number: int) -> str:
    """Normalize fixed-width lottery values in memory without editing the source CSV."""
    raw_value = str(value).strip()
    if not raw_value.isdigit():
        raise ValueError(f"Row {row_number}: invalid {column} value {raw_value!r}; digits only")
    if len(raw_value) > length:
        raise ValueError(f"Row {row_number}: invalid {column} value {raw_value!r}; max {length} digits")
    return raw_value.zfill(length)


def load_history(history_path: Path) -> list[dict[str, str]]:
    """Read lottery history as strings and normalize values in memory."""
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            cleaned = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
            cleaned["date"] = normalize_date(cleaned["date"])
            for column, length in FIELD_LENGTHS.items():
                cleaned[column] = normalize_number(cleaned[column], length, column, row_number)
            rows.append(cleaned)

    if not rows:
        raise ValueError("Input CSV has no data rows.")
    return sort_rows(rows, descending=True)


def sort_rows(rows: list[dict[str, str]], descending: bool) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row["date"], reverse=descending)


def recent_weight(index: int) -> int:
    if index < 10:
        return 5
    if index < 50:
        return 2
    return 1


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {key: 0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def ranked_scores(scores: dict[str, float], limit: int, value_key: str = "number", score_key: str = "score") -> list[dict[str, Any]]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{value_key: value, score_key: score, "rank": rank} for rank, (value, score) in enumerate(ranked, start=1)]


def ranked_counts(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"rank": rank, "number": value, "count": count} for rank, (value, count) in enumerate(ranked, start=1)]


def observed_values(rows: list[dict[str, str]], value_type: str) -> list[str]:
    if value_type == "last2":
        return sorted({row["last2"] for row in rows})
    values: set[str] = set()
    for row in rows:
        for column in THREE_DIGIT_COLUMNS:
            values.add(row[column])
    return sorted(values)


def frequency_score_map(rows: list[dict[str, str]], value_type: str) -> dict[str, float]:
    counter: Counter[str] = Counter()
    if value_type == "last2":
        counter.update(row["last2"] for row in rows)
    else:
        for row in rows:
            counter.update(row[column] for column in THREE_DIGIT_COLUMNS)
    return dict(counter)


def recent_score_map(rows: list[dict[str, str]], value_type: str) -> dict[str, float]:
    recent_rows = sort_rows(rows, descending=True)
    scores: dict[str, float] = {}
    for index, row in enumerate(recent_rows):
        weight = recent_weight(index)
        if value_type == "last2":
            scores[row["last2"]] = scores.get(row["last2"], 0) + weight
        else:
            for column in THREE_DIGIT_COLUMNS:
                value = row[column]
                scores[value] = scores.get(value, 0) + weight
    return scores


def position_score_map(rows: list[dict[str, str]], value_type: str, candidates: list[str]) -> dict[str, float]:
    if value_type == "last2":
        tens = Counter(row["last2"][0] for row in rows)
        ones = Counter(row["last2"][1] for row in rows)
        return {value: tens[value[0]] + ones[value[1]] for value in candidates}

    hundreds: Counter[str] = Counter()
    tens3: Counter[str] = Counter()
    ones3: Counter[str] = Counter()
    for row in rows:
        for column in THREE_DIGIT_COLUMNS:
            value = row[column]
            hundreds[value[0]] += 1
            tens3[value[1]] += 1
            ones3[value[2]] += 1
    return {value: hundreds[value[0]] + tens3[value[1]] + ones3[value[2]] for value in candidates}


def hot_score_map(rows: list[dict[str, str]], value_type: str, recent_limit: int = 20) -> dict[str, float]:
    recent_rows = sort_rows(rows, descending=True)[:recent_limit]
    return frequency_score_map(recent_rows, value_type)


def frequency_suggestions(rows: list[dict[str, str]], limit: int = 10) -> dict[str, Any]:
    return {
        "method": METHOD,
        "source_rows": len(rows),
        "suggested_last2": ranked_counts(Counter(frequency_score_map(rows, "last2")), limit),
        "suggested_3digit": ranked_counts(Counter(frequency_score_map(rows, "3digit")), limit),
        "note": SAFETY_NOTE,
    }


def weighted_recent_suggestions(rows: list[dict[str, str]], limit: int = 10) -> dict[str, Any]:
    return {
        "method": WEIGHTED_METHOD,
        "source_rows": len(rows),
        "suggested_last2": ranked_scores(recent_score_map(rows, "last2"), limit),
        "suggested_3digit": ranked_scores(recent_score_map(rows, "3digit"), limit),
        "note": SAFETY_NOTE,
    }


def hybrid_score_map(rows: list[dict[str, str]], value_type: str) -> dict[str, float]:
    candidates = observed_values(rows, value_type)
    freq = normalize_scores({value: frequency_score_map(rows, value_type).get(value, 0) for value in candidates})
    recent = normalize_scores({value: recent_score_map(rows, value_type).get(value, 0) for value in candidates})
    position = normalize_scores(position_score_map(rows, value_type, candidates))
    hot = normalize_scores({value: hot_score_map(rows, value_type).get(value, 0) for value in candidates})
    return {
        value: (freq.get(value, 0) * 0.40)
        + (recent.get(value, 0) * 0.30)
        + (position.get(value, 0) * 0.20)
        + (hot.get(value, 0) * 0.10)
        for value in candidates
    }


def hybrid_suggestions(rows: list[dict[str, str]], limit: int = 20) -> dict[str, Any]:
    return {
        "method": HYBRID_METHOD,
        "source_rows": len(rows),
        "suggested_last2": ranked_scores(hybrid_score_map(rows, "last2"), limit, score_key="final_score"),
        "suggested_3digit": ranked_scores(hybrid_score_map(rows, "3digit"), limit, score_key="final_score"),
        "note": SAFETY_NOTE,
    }


def generate_suggestions(history_path: Path, method: str = METHOD, top_n: int = 10) -> dict[str, Any]:
    return generate_suggestions_from_rows(load_history(history_path), method, top_n)


def generate_suggestions_from_rows(rows: list[dict[str, str]], method: str = METHOD, top_n: int = 10) -> dict[str, Any]:
    if method == METHOD:
        return frequency_suggestions(rows, top_n)
    if method == WEIGHTED_METHOD:
        return weighted_recent_suggestions(rows, top_n)
    if method == HYBRID_METHOD:
        return hybrid_suggestions(rows, top_n)
    raise ValueError(f"Unsupported method: {method}")


def digit_position_analysis(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    last2_positions = {"tens": Counter(row["last2"][0] for row in rows), "ones": Counter(row["last2"][1] for row in rows)}
    three_positions = {"hundreds": Counter(), "tens": Counter(), "ones": Counter()}
    for row in rows:
        for column in THREE_DIGIT_COLUMNS:
            value = row[column]
            three_positions["hundreds"][value[0]] += 1
            three_positions["tens"][value[1]] += 1
            three_positions["ones"][value[2]] += 1
    return {"last2": position_rows(last2_positions, "last2"), "three_digit": position_rows(three_positions, "3digit")}


def position_rows(position_counts: dict[str, Counter[str]], group_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, counter in position_counts.items():
        for digit in [str(value) for value in range(10)]:
            rows.append({"group": group_name, "position": position, "digit": digit, "count": counter[digit]})
    return rows


def hot_cold_numbers(rows: list[dict[str, str]], recent_limit: int = 20, result_limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    recent_rows = sort_rows(rows, descending=True)[:recent_limit]
    oldest_rows = sort_rows(rows, descending=False)
    return {
        "hot_last2": ranked_counts(Counter(frequency_score_map(recent_rows, "last2")), result_limit),
        "hot_3digit": ranked_counts(Counter(frequency_score_map(recent_rows, "3digit")), result_limit),
        "cold_last2": cold_values(oldest_rows, "last2", result_limit),
        "cold_3digit": cold_3digit_values(oldest_rows, result_limit),
    }


def cold_values(oldest_rows: list[dict[str, str]], column: str, limit: int) -> list[dict[str, Any]]:
    last_seen: dict[str, int] = {}
    total_rows = len(oldest_rows)
    for index, row in enumerate(oldest_rows):
        last_seen[row[column]] = index
    ranked = sorted(last_seen.items(), key=lambda item: (item[1], item[0]))[:limit]
    return [
        {"rank": rank, "number": value, "rounds_since_seen": total_rows - 1 - last_index}
        for rank, (value, last_index) in enumerate(ranked, start=1)
    ]


def cold_3digit_values(oldest_rows: list[dict[str, str]], limit: int) -> list[dict[str, Any]]:
    last_seen: dict[str, int] = {}
    total_rows = len(oldest_rows)
    for index, row in enumerate(oldest_rows):
        for column in THREE_DIGIT_COLUMNS:
            last_seen[row[column]] = index
    ranked = sorted(last_seen.items(), key=lambda item: (item[1], item[0]))[:limit]
    return [
        {"rank": rank, "number": value, "rounds_since_seen": total_rows - 1 - last_index}
        for rank, (value, last_index) in enumerate(ranked, start=1)
    ]


def phase3_analysis(history_path: Path, top_n: int = 10) -> dict[str, Any]:
    rows = load_history(history_path)
    return {
        "frequency": frequency_suggestions(rows, top_n),
        "weighted_recent": weighted_recent_suggestions(rows, top_n),
        "hybrid": hybrid_suggestions(rows, max(top_n, 20)),
        "positions": digit_position_analysis(rows),
        "hot_cold": hot_cold_numbers(rows),
        "source_rows": len(rows),
        "latest_draw": sort_rows(rows, descending=True)[0]["date"],
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "note": SAFETY_NOTE,
    }


def run_backtest(rows: list[dict[str, str]], top_n: int = 20, rolling_span: int | None = None) -> dict[str, Any]:
    chronological = sort_rows(rows, descending=False)
    details: list[dict[str, Any]] = []

    for method in SUPPORTED_METHODS:
        for index in range(1, len(chronological)):
            prior_rows = chronological[:index]
            if rolling_span and rolling_span > 0:
                prior_rows = prior_rows[-rolling_span:]
            actual = chronological[index]
            suggestions = generate_suggestions_from_rows(prior_rows, method, top_n)
            suggested_last2 = [item["number"] for item in suggestions["suggested_last2"]]
            suggested_3digit = [item["number"] for item in suggestions["suggested_3digit"]]
            actual_three = sorted({actual[column] for column in THREE_DIGIT_COLUMNS})
            last2_hit = actual["last2"] in suggested_last2
            three_hit_values = [value for value in actual_three if value in suggested_3digit]
            three_hit = bool(three_hit_values)
            hit_ranks: list[int] = []
            if last2_hit:
                hit_ranks.append(suggested_last2.index(actual["last2"]) + 1)
            for value in three_hit_values:
                hit_ranks.append(suggested_3digit.index(value) + 1)
            rank_hit = min(hit_ranks) if hit_ranks else ""
            details.append(
                {
                    "draw_date": actual["date"],
                    "method": method,
                    "suggested_last2": ",".join(suggested_last2),
                    "actual_last2": actual["last2"],
                    "hit_last2": "1" if last2_hit else "0",
                    "suggested_3digit": ",".join(suggested_3digit),
                    "actual_3digit": ",".join(actual_three),
                    "hit_3digit": "1" if three_hit else "0",
                    "rank_hit": str(rank_hit),
                }
            )

    summary = summarize_backtest(details)
    return {"summary": summary, "details": details, "note": SAFETY_NOTE}


def summarize_backtest(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in SUPPORTED_METHODS:
        method_rows = [row for row in details if row["method"] == method]
        tested_rounds = len(method_rows)
        last2_hit_count = sum(int(row["hit_last2"]) for row in method_rows)
        three_hit_count = sum(int(row["hit_3digit"]) for row in method_rows)
        rank_values = [int(row["rank_hit"]) for row in method_rows if str(row["rank_hit"]).isdigit()]
        recent_rows = method_rows[-20:]
        recent_hits = sum(1 for row in recent_rows if row["hit_last2"] == "1" or row["hit_3digit"] == "1")
        summaries.append(
            {
                "method": method,
                "tested_rounds": tested_rounds,
                "last2_hit_count": last2_hit_count,
                "last2_hit_rate": last2_hit_count / tested_rounds if tested_rounds else 0,
                "3digit_hit_count": three_hit_count,
                "3digit_hit_rate": three_hit_count / tested_rounds if tested_rounds else 0,
                "avg_rank_hit": sum(rank_values) / len(rank_values) if rank_values else 0,
                "recent_20_round_hit_rate": recent_hits / len(recent_rows) if recent_rows else 0,
                "note": SAFETY_NOTE,
            }
        )
    return summaries


def rolling_accuracy_trend(details: list[dict[str, Any]], span: int = 20) -> list[dict[str, Any]]:
    trend: list[dict[str, Any]] = []
    for method in SUPPORTED_METHODS:
        rows = [row for row in details if row["method"] == method]
        for index, row in enumerate(rows):
            slice_rows = rows[max(0, index - span + 1) : index + 1]
            hits = sum(1 for item in slice_rows if item["hit_last2"] == "1" or item["hit_3digit"] == "1")
            trend.append(
                {
                    "draw_date": row["draw_date"],
                    "method": method,
                    "rolling_hit_rate": hits / len(slice_rows) if slice_rows else 0,
                }
            )
    return trend


def backtest_methods(
    history_path: Path,
    output_path: Path,
    backup_dir: Path,
    detail_csv_path: Path | None = None,
    top_n: int = 20,
    rolling_span: int | None = None,
) -> dict[str, Any]:
    rows = load_history(history_path)
    result = run_backtest(rows, top_n=top_n, rolling_span=rolling_span)
    backup_file(output_path, backup_dir, "backtest_result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}__tmp{output_path.suffix}")
    export_backtest_xlsx(temp_path, result)
    shutil.move(str(temp_path), str(output_path))

    if detail_csv_path is not None:
        write_backtest_detail_csv(detail_csv_path, backup_dir, result["details"])
    return result


def write_backtest_detail_csv(path: Path, backup_dir: Path, details: list[dict[str, Any]]) -> None:
    backup_file(path, backup_dir, "backtest_detail")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}__tmp{path.suffix}")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BACKTEST_DETAIL_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in BACKTEST_DETAIL_COLUMNS} for row in details])
    shutil.move(str(temp_path), str(path))


def ensure_prediction_history(history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        return
    with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_HISTORY_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()


def read_prediction_history(history_path: Path) -> list[dict[str, str]]:
    ensure_prediction_history(history_path)
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in PREDICTION_HISTORY_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing prediction history column(s): {', '.join(missing)}")
        return [{column: (row.get(column) or "").strip() for column in PREDICTION_HISTORY_COLUMNS} for row in reader]


def backup_file(path: Path, backup_dir: Path, label: str) -> Path | None:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}__backup__{stamp}__{label}{path.suffix}"
    shutil.copy2(path, backup_path)
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError(f"Backup failed: {backup_path}")
    return backup_path


def save_analysis(prediction_history_path: Path, backup_dir: Path, target_draw: str, suggestions: dict[str, Any]) -> Path | None:
    normalized_target_draw = normalize_date(target_draw)
    ensure_prediction_history(prediction_history_path)
    backup_path = backup_file(prediction_history_path, backup_dir, "save_analysis")
    existing_rows = read_prediction_history(prediction_history_path)
    new_record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_draw": normalized_target_draw,
        "method": suggestions["method"],
        "suggested_last2": ",".join(item["number"] for item in suggestions["suggested_last2"]),
        "suggested_3digit": ",".join(item["number"] for item in suggestions["suggested_3digit"]),
        "source_rows": str(suggestions["source_rows"]),
        "note": suggestions["note"],
    }
    output_rows = existing_rows + [new_record]
    temp_path = prediction_history_path.with_name(f"{prediction_history_path.stem}__tmp{prediction_history_path.suffix}")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_HISTORY_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)
    shutil.move(str(temp_path), str(prediction_history_path))
    return backup_path


def export_backtest_xlsx(path: Path, result: dict[str, Any]) -> None:
    sheets = [("Summary", result["summary"]), ("Details", result["details"]), ("Rolling_Trend", rolling_accuracy_trend(result["details"]))]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    files: dict[str, str] = {
        "[Content_Types].xml": content_types_xml(len(sheets)),
        "_rels/.rels": root_rels_xml(),
        "docProps/app.xml": app_props_xml(),
        "docProps/core.xml": core_props_xml(created),
        "xl/workbook.xml": workbook_xml([name for name, _rows in sheets]),
        "xl/_rels/workbook.xml.rels": workbook_rels_xml(len(sheets)),
        "xl/styles.xml": styles_xml(),
    }
    for index, (_name, rows) in enumerate(sheets, start=1):
        files[f"xl/worksheets/sheet{index}.xml"] = sheet_xml(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def sheet_xml(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0].keys()) if rows else ["note"]
    if not rows:
        rows = [{"note": SAFETY_NOTE}]
    matrix = [headers] + [[row.get(header, "") for header in headers] for row in rows]
    row_xml: list[str] = []
    for row_index, row in enumerate(matrix, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{col_name(col_index)}{row_index}"
            style = "4" if row_index == 1 else "1"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                style = "3" if isinstance(value, float) and 0 <= value <= 1 else "2"
                cells.append(f'<c r="{ref}" s="{style}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{escape(str(value))}</t></is></c>')
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    last_col = col_name(len(headers))
    widths = "".join(f'<col min="{i}" max="{i}" width="22" customWidth="1"/>' for i in range(1, len(headers) + 1))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<cols>{widths}</cols>"
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        f'<autoFilter ref="A1:{last_col}{len(matrix)}"/>'
        "</worksheet>"
    )


def col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def content_types_xml(sheet_count: int) -> str:
    sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheets}"
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def app_props_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application></Properties>'
    )


def core_props_xml(created: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified></cp:coreProperties>'
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    sheets = "".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{sheets}"
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="2"><numFmt numFmtId="164" formatCode="@"/><numFmt numFmtId="165" formatCode="0.00%"/></numFmts>'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="164" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyNumberFormat="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'
    )


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    analysis = phase3_analysis(base_dir / "data" / "lottery_history.csv", top_n=20)
    print(analysis["hybrid"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
