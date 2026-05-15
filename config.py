"""Runtime configuration for the Lottery Statistics Dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from constants import BACKUP_RETENTION_LIMIT


@dataclass(frozen=True)
class AppConfig:
    default_top_n: int = 20
    min_top_n: int = 5
    max_top_n: int = 20
    top_n_step: int = 1
    rolling_spans: tuple[int, ...] = (0, 20, 50, 100)
    subprocess_timeout_seconds: int = 120
    backup_retention_limit: int = BACKUP_RETENTION_LIMIT


APP_CONFIG = AppConfig()
