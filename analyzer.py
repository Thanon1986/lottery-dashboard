#!/usr/bin/env python3
"""
Analyze lottery history from data/lottery_history.csv.

This script is intentionally descriptive, not predictive. It validates the CSV
schema, reads lottery values as strings, counts historical frequencies, backs up
an existing output workbook, writes through a temp workspace, and records a run
log. It never edits lottery_history.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "1.0.0"
AGENT_NAME = "lottery_history_analyzer"
WORKFLOW_NAME = "lottery_frequency_analysis"
SAFETY_NOTE = "Historical statistical analysis only. This system does not predict or guarantee lottery outcomes."

REQUIRED_COLUMNS = [
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

THREE_DIGIT_COLUMNS = ["front3_1", "front3_2", "back3_1", "back3_2"]

FIELD_LENGTHS = {
    "first_prize": 6,
    "last2": 2,
    "front3_1": 3,
    "front3_2": 3,
    "back3_1": 3,
    "back3_2": 3,
}


@dataclass
class RunConfig:
    input_csv: Path
    output_xlsx: Path
    data_quality_xlsx: Path
    backup_dir: Path
    log_dir: Path
    temp_dir: Path


@dataclass
class RunLog:
    run_id: str
    start_timestamp: str
    end_timestamp: str | None = None
    agent_name: str = AGENT_NAME
    workflow_name: str = WORKFLOW_NAME
    script_name: str = "analyzer.py"
    script_version: str = SCRIPT_VERSION
    input_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    backup_files: list[str] = field(default_factory=list)
    temp_workspace: str | None = None
    records_processed: int = 0
    rows_updated: int = 0
    validation_summary: str = "not_run"
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cleanup_result: str = "not_run"
    final_status: str = "not_run"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze historical lottery frequency.")
    parser.add_argument("--input", default="data/lottery_history.csv")
    parser.add_argument("--output", default="output/stat_summary.xlsx")
    parser.add_argument("--data-quality-output", default="output/data_quality_report.xlsx")
    parser.add_argument("--backup-dir", default="backup")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--temp-dir", default="temp")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        input_csv=Path(args.input).resolve(),
        output_xlsx=Path(args.output).resolve(),
        data_quality_xlsx=Path(args.data_quality_output).resolve(),
        backup_dir=Path(args.backup_dir).resolve(),
        log_dir=Path(args.log_dir).resolve(),
        temp_dir=Path(args.temp_dir).resolve(),
    )


def validate_columns(fieldnames: list[str] | None) -> list[str]:
    """Return missing required columns from the CSV header."""
    if fieldnames is None:
        return REQUIRED_COLUMNS.copy()
    return [column for column in REQUIRED_COLUMNS if column not in fieldnames]


def load_history(input_csv: Path) -> list[dict[str, str]]:
    """Read CSV values as strings and validate schema/value shapes."""
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = validate_columns(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader, start=2):
            cleaned = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
            cleaned["date"] = normalize_date(cleaned["date"])
            for column, length in FIELD_LENGTHS.items():
                cleaned[column] = normalize_number(cleaned[column], length, column, index)
            validate_row(cleaned, index)
            rows.append(cleaned)

    if not rows:
        raise ValueError("Input CSV has no data rows.")
    return rows


def data_quality_checks(input_csv: Path) -> dict[str, list[dict[str, Any]]]:
    """Validate source data quality without modifying the CSV."""
    issues: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    seen_dates: dict[str, int] = {}
    total_rows = 0

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = validate_columns(reader.fieldnames)
        for column in missing_columns:
            issues.append({"severity": "critical", "row": "header", "column": column, "issue": "missing required column", "value": ""})

        if missing_columns:
            summary.append({"metric": "total_rows", "value": 0})
            summary.append({"metric": "issue_count", "value": len(issues)})
            summary.append({"metric": "note", "value": SAFETY_NOTE})
            return {"summary": summary, "issues": issues}

        for row_number, row in enumerate(reader, start=2):
            total_rows += 1
            raw_date = (row.get("date") or "").strip()
            if not raw_date:
                issues.append({"severity": "critical", "row": row_number, "column": "date", "issue": "missing value", "value": raw_date})
            else:
                try:
                    normalized = normalize_date(raw_date)
                    if normalized in seen_dates:
                        issues.append({"severity": "critical", "row": row_number, "column": "date", "issue": "duplicate date", "value": raw_date})
                    seen_dates[normalized] = row_number
                except ValueError:
                    issues.append({"severity": "critical", "row": row_number, "column": "date", "issue": "invalid date format", "value": raw_date})

            for column, length in FIELD_LENGTHS.items():
                value = (row.get(column) or "").strip()
                if not value:
                    issues.append({"severity": "critical", "row": row_number, "column": column, "issue": "missing value", "value": value})
                elif not value.isdigit():
                    issues.append({"severity": "critical", "row": row_number, "column": column, "issue": "non-digit characters", "value": value})
                elif len(value) != length:
                    severity = "critical" if len(value) > length else "warning"
                    issues.append({"severity": severity, "row": row_number, "column": column, "issue": f"invalid digit length; expected {length}", "value": value})

    summary = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "issue_count", "value": len(issues)},
        {"metric": "critical_count", "value": sum(1 for issue in issues if issue["severity"] == "critical")},
        {"metric": "warning_count", "value": sum(1 for issue in issues if issue["severity"] == "warning")},
        {"metric": "note", "value": SAFETY_NOTE},
    ]
    return {"summary": summary, "issues": issues or [{"severity": "ok", "row": "", "column": "", "issue": "no data quality issues", "value": ""}]}


def normalize_date(value: str) -> str:
    """Normalize supported dates to YYYY-MM-DD without changing the source file."""
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


def normalize_number(value: str, length: int, column: str, row_number: int) -> str:
    """Normalize fixed-width lottery values in memory without editing the CSV."""
    raw_value = str(value).strip()
    if not raw_value.isdigit():
        raise ValueError(f"Row {row_number}: invalid {column} value {raw_value!r}; digits only")
    if len(raw_value) > length:
        raise ValueError(f"Row {row_number}: invalid {column} value {raw_value!r}; max {length} digits")
    return raw_value.zfill(length)


def validate_row(row: dict[str, str], row_number: int) -> None:
    patterns = {
        "date": r"^\d{4}-\d{2}-\d{2}$",
        "first_prize": r"^\d{6}$",
        "last2": r"^\d{2}$",
        "front3_1": r"^\d{3}$",
        "front3_2": r"^\d{3}$",
        "back3_1": r"^\d{3}$",
        "back3_2": r"^\d{3}$",
    }
    for column, pattern in patterns.items():
        value = row[column]
        if not re.fullmatch(pattern, value):
            raise ValueError(f"Row {row_number}: invalid {column} value {value!r}")


def analyze_digit_frequency(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Count digit 0-9 frequency across all required lottery number columns."""
    by_digit: dict[str, Counter[str]] = {str(digit): Counter() for digit in range(10)}
    total_digits = 0

    for row in rows:
        for column in NUMBER_COLUMNS:
            value = row[column]
            for digit in value:
                by_digit[digit][column] += 1
                by_digit[digit]["total"] += 1
                total_digits += 1

    results: list[dict[str, Any]] = []
    for digit in [str(value) for value in range(10)]:
        counter = by_digit[digit]
        item: dict[str, Any] = {"digit": digit, "total": counter["total"]}
        for column in NUMBER_COLUMNS:
            item[column] = counter[column]
        item["share"] = counter["total"] / total_digits if total_digits else 0
        results.append(item)
    return results


def analyze_top_last2(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    """Return top historical last2 values."""
    counts = Counter(row["last2"] for row in rows)
    return ranked_counter(counts, len(rows), limit, "last2")


def analyze_top_3digits(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    """Return top historical 3-digit values across front/back 3-digit columns."""
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    for row in rows:
        for column in THREE_DIGIT_COLUMNS:
            value = row[column]
            counts[value] += 1
            source_counts.setdefault(value, Counter())[column] += 1

    total_values = len(rows) * len(THREE_DIGIT_COLUMNS)
    ranked = ranked_counter(counts, total_values, limit, "number_3digit")
    for item in ranked:
        source = source_counts[item["number_3digit"]]
        item["front3_1"] = source["front3_1"]
        item["front3_2"] = source["front3_2"]
        item["back3_1"] = source["back3_1"]
        item["back3_2"] = source["back3_2"]
    return ranked


def ranked_counter(counter: Counter[str], denominator: int, limit: int, value_key: str) -> list[dict[str, Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "rank": index,
            value_key: value,
            "count": count,
            "share": count / denominator if denominator else 0,
        }
        for index, (value, count) in enumerate(ranked, start=1)
    ]


def build_analysis(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "digit_frequency": analyze_digit_frequency(rows),
        "top_last2": analyze_top_last2(rows),
        "top_3digits": analyze_top_3digits(rows),
    }


def backup_path(path: Path, config: RunConfig, run_id: str, stamp: str, log: RunLog) -> None:
    if not path.exists():
        log.warnings.append(f"No existing {path.name} found; backup was not required.")
        return

    backup_name = f"{path.stem}__backup__{stamp}__{run_id}{path.suffix}"
    backup_path = config.backup_dir / backup_name
    shutil.copy2(path, backup_path)
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise RuntimeError(f"Backup verification failed: {backup_path}")
    log.backup_files.append(str(backup_path))


def backup_existing_output(config: RunConfig, run_id: str, stamp: str, log: RunLog) -> None:
    backup_path(config.output_xlsx, config, run_id, stamp, log)
    backup_path(config.data_quality_xlsx, config, run_id, stamp, log)


def col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def make_text(value: Any) -> dict[str, Any]:
    return {"value": str(value), "type": "text", "style": 1}


def make_number(value: Any) -> dict[str, Any]:
    return {"value": value, "type": "number", "style": 2}


def make_percent(value: Any) -> dict[str, Any]:
    return {"value": value, "type": "number", "style": 3}


def make_header(value: Any) -> dict[str, Any]:
    return {"value": str(value), "type": "text", "style": 4}


def cell_xml(cell_ref: str, cell: dict[str, Any]) -> str:
    style = cell.get("style", 0)
    value = cell.get("value", "")
    if cell.get("type") == "number":
        return f'<c r="{cell_ref}" s="{style}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr" s="{style}"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(rows: list[list[dict[str, Any]]], widths: list[float]) -> str:
    col_xml = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells = [
            cell_xml(f"{col_name(col_index)}{row_index}", cell)
            for col_index, cell in enumerate(row, start=1)
        ]
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    dimension_ref = f"A1:{col_name(max(len(widths), 1))}{max(len(rows), 1)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension_ref}"/>'
        f"<cols>{col_xml}</cols>"
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        f'<autoFilter ref="{dimension_ref}"/>'
        '</worksheet>'
    )


def build_summary_sheet(rows: list[dict[str, str]], analysis: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [
        [make_header("metric"), make_header("value")],
        [make_text("source_rows"), make_number(len(rows))],
        [make_text("digit_frequency_digits_counted"), make_number(sum(item["total"] for item in analysis["digit_frequency"]))],
        [make_text("last2_values_counted"), make_number(len(rows))],
        [make_text("three_digit_values_counted"), make_number(len(rows) * len(THREE_DIGIT_COLUMNS))],
        [make_text("note"), make_text(SAFETY_NOTE)],
    ]


def build_digit_sheet(analysis: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    headers = ["digit", "total", *NUMBER_COLUMNS, "share"]
    rows = [[make_header(header) for header in headers]]
    for item in analysis["digit_frequency"]:
        rows.append(
            [
                make_text(item["digit"]),
                make_number(item["total"]),
                *[make_number(item[column]) for column in NUMBER_COLUMNS],
                make_percent(item["share"]),
            ]
        )
    return rows


def build_top_last2_sheet(analysis: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    rows = [[make_header("rank"), make_header("last2"), make_header("count"), make_header("share")]]
    for item in analysis["top_last2"]:
        rows.append([make_number(item["rank"]), make_text(item["last2"]), make_number(item["count"]), make_percent(item["share"])])
    return rows


def build_top_3digit_sheet(analysis: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    headers = ["rank", "number_3digit", "count", "share", *THREE_DIGIT_COLUMNS]
    rows = [[make_header(header) for header in headers]]
    for item in analysis["top_3digits"]:
        rows.append(
            [
                make_number(item["rank"]),
                make_text(item["number_3digit"]),
                make_number(item["count"]),
                make_percent(item["share"]),
                *[make_number(item[column]) for column in THREE_DIGIT_COLUMNS],
            ]
        )
    return rows


def export_summary_xlsx(output_path: Path, rows: list[dict[str, str]], analysis: dict[str, list[dict[str, Any]]]) -> None:
    sheets = [
        ("Summary", build_summary_sheet(rows, analysis), [32, 88]),
        ("Digit_Frequency", build_digit_sheet(analysis), [10, 12, 14, 14, 14, 12, 14, 14, 12]),
        ("Top_Last2", build_top_last2_sheet(analysis), [10, 12, 12, 12]),
        ("Top_3Digit", build_top_3digit_sheet(analysis), [10, 16, 12, 12, 14, 14, 14, 14]),
    ]

    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    files: dict[str, str] = {
        "[Content_Types].xml": build_content_types(len(sheets)),
        "_rels/.rels": build_root_rels(),
        "docProps/app.xml": build_app_props(),
        "docProps/core.xml": build_core_props(created),
        "xl/workbook.xml": build_workbook_xml([sheet[0] for sheet in sheets]),
        "xl/_rels/workbook.xml.rels": build_workbook_rels(len(sheets)),
        "xl/styles.xml": build_styles_xml(),
    }

    for index, (_name, matrix, widths) in enumerate(sheets, start=1):
        files[f"xl/worksheets/sheet{index}.xml"] = sheet_xml(matrix, widths)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def export_data_quality_xlsx(output_path: Path, quality: dict[str, list[dict[str, Any]]]) -> None:
    sheets = [
        ("Summary", quality_rows_to_matrix(quality["summary"])),
        ("Issues", quality_rows_to_matrix(quality["issues"])),
    ]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    files: dict[str, str] = {
        "[Content_Types].xml": build_content_types(len(sheets)),
        "_rels/.rels": build_root_rels(),
        "docProps/app.xml": build_app_props(),
        "docProps/core.xml": build_core_props(created),
        "xl/workbook.xml": build_workbook_xml([sheet[0] for sheet in sheets]),
        "xl/_rels/workbook.xml.rels": build_workbook_rels(len(sheets)),
        "xl/styles.xml": build_styles_xml(),
    }
    for index, (_name, matrix) in enumerate(sheets, start=1):
        files[f"xl/worksheets/sheet{index}.xml"] = sheet_xml(matrix, [22] * max(len(matrix[0]), 1))

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def quality_rows_to_matrix(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    headers = list(rows[0].keys()) if rows else ["note"]
    matrix: list[list[dict[str, Any]]] = [[make_header(header) for header in headers]]
    for row in rows:
        matrix.append([make_number(value) if isinstance(value, int) else make_text(value) for value in [row.get(header, "") for header in headers]])
    return matrix


def build_content_types(sheet_count: int) -> str:
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{sheet_overrides}"
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '</Types>'
    )


def build_root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )


def build_app_props() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<Application>Codex</Application>'
        '</Properties>'
    )


def build_core_props(created: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Codex</dc:creator>'
        '<cp:lastModifiedBy>Codex</cp:lastModifiedBy>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>'
        '</cp:coreProperties>'
    )


def build_workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets>"
        '</workbook>'
    )


def build_workbook_rels(sheet_count: int) -> str:
    sheet_rels = "".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{sheet_rels}"
        f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '</Relationships>'
    )


def build_styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="2">'
        '<numFmt numFmtId="164" formatCode="@"/>'
        '<numFmt numFmtId="165" formatCode="0.00%"/>'
        '</numFmts>'
        '<fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="5">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="164" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyNumberFormat="1"/>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )


def verify_output_xlsx(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Output workbook missing or empty: {path}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {
            "xl/workbook.xml",
            "xl/styles.xml",
            "xl/worksheets/sheet1.xml",
            "xl/worksheets/sheet2.xml",
            "xl/worksheets/sheet3.xml",
            "xl/worksheets/sheet4.xml",
        }
        missing = required - names
        if missing:
            raise ValueError(f"Output workbook missing part(s): {sorted(missing)}")
        styles = archive.read("xl/styles.xml").decode("utf-8")
        if 'formatCode="@"' not in styles:
            raise ValueError("Plain Text format was not found in workbook styles.")


def write_log(config: RunConfig, log: RunLog, stamp: str) -> Path:
    log.end_timestamp = iso_now()
    log_path = config.log_dir / f"{AGENT_NAME}__run__{stamp}__{log.run_id}.json"
    log_path.write_text(json.dumps(asdict(log), ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path


def run(config: RunConfig) -> tuple[int, RunLog]:
    stamp = now_stamp()
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    log = RunLog(run_id=run_id, start_timestamp=iso_now())
    log.input_files.append(str(config.input_csv))
    temp_workspace: Path | None = None

    try:
        for folder in [config.output_xlsx.parent, config.data_quality_xlsx.parent, config.backup_dir, config.log_dir, config.temp_dir]:
            folder.mkdir(parents=True, exist_ok=True)
        if not config.input_csv.exists():
            raise FileNotFoundError(f"Missing input CSV: {config.input_csv}")

        temp_workspace = config.temp_dir / f"run_{stamp}_{run_id}"
        temp_workspace.mkdir(parents=True, exist_ok=False)
        log.temp_workspace = str(temp_workspace)

        backup_existing_output(config, run_id, stamp, log)

        quality = data_quality_checks(config.input_csv)
        temp_quality = temp_workspace / config.data_quality_xlsx.name
        export_data_quality_xlsx(temp_quality, quality)
        shutil.copy2(temp_quality, config.data_quality_xlsx)
        log.output_files.append(str(config.data_quality_xlsx))

        critical_quality = [issue for issue in quality["issues"] if issue.get("severity") == "critical"]
        if critical_quality:
            log.critical_errors.append(f"Data quality critical issues found: {len(critical_quality)}")
            log.validation_summary = "failed"
            log.final_status = "failed"
            return 1, log
        warning_quality = [issue for issue in quality["issues"] if issue.get("severity") == "warning"]
        if warning_quality:
            log.warnings.append(f"Data quality warnings found: {len(warning_quality)}")

        rows = load_history(config.input_csv)
        log.records_processed = len(rows)
        analysis = build_analysis(rows)

        temp_output = temp_workspace / config.output_xlsx.name
        export_summary_xlsx(temp_output, rows, analysis)
        verify_output_xlsx(temp_output)

        shutil.copy2(temp_output, config.output_xlsx)
        verify_output_xlsx(config.output_xlsx)

        log.output_files.append(str(config.output_xlsx))
        log.rows_updated = (
            len(analysis["digit_frequency"])
            + len(analysis["top_last2"])
            + len(analysis["top_3digits"])
            + 5
        )
        log.validation_summary = "passed"
        log.final_status = "success_with_warnings" if log.warnings else "success"
        return 0, log
    except Exception as exc:
        log.critical_errors.append(f"{type(exc).__name__}: {exc}")
        log.validation_summary = "failed"
        log.final_status = "failed"
        return 1, log
    finally:
        if temp_workspace is not None:
            try:
                shutil.rmtree(temp_workspace)
                log.cleanup_result = "success"
            except OSError as exc:
                log.warnings.append(f"Temp cleanup failed: {exc}")
                log.cleanup_result = "warning_cleanup_failed"
        else:
            log.cleanup_result = "skipped_no_temp_workspace"
        try:
            write_log(config, log, stamp)
        except Exception:
            pass


def main() -> int:
    config = resolve_config(parse_args())
    exit_code, log = run(config)
    summary = {
        "status": log.final_status,
        "records_processed": log.records_processed,
        "output_files": log.output_files,
        "backup_files": log.backup_files,
        "critical_errors": log.critical_errors,
        "warnings": log.warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
