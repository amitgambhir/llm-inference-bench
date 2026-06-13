"""Background job runner for benchmark subprocess execution.

run_job() is designed to be called via FastAPI BackgroundTasks.  It blocks
until the subprocess finishes and then persists the outcome.  The session_factory
argument lets it open its own DB connection — safe to call after the request
session has already been closed.
"""
from __future__ import annotations

import shlex
import subprocess
from datetime import datetime, timezone
from typing import Callable


def run_job(
    run_id: int,
    command: str,
    result_path: str,
    session_factory: Callable,
) -> None:
    """Execute a benchmark command and persist the outcome to the DB."""
    session = session_factory()
    try:
        from api.db import BenchmarkRunRow

        run = session.get(BenchmarkRunRow, run_id)
        if run is not None:
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            session.commit()
    finally:
        session.close()

    try:
        proc = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        status = "done" if proc.returncode == 0 else "failed"
        stdout = proc.stdout or ""
        if proc.stderr:
            stdout = (stdout + "\n" + proc.stderr).strip()
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        status = "failed"
        stdout = "Timed out after 2 hours."
        exit_code = -1
    except Exception as exc:
        status = "failed"
        stdout = str(exc)
        exit_code = -1

    session = session_factory()
    try:
        from api.db import BenchmarkRunRow

        run = session.get(BenchmarkRunRow, run_id)
        if run is not None:
            run.status = status
            run.stdout = stdout.strip()
            run.exit_code = exit_code
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
    finally:
        session.close()
