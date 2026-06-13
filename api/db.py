"""SQLAlchemy ORM models and session-factory factory for the capacity planner API.

Postgres-ready: swap the DATABASE_URL and remove connect_args to migrate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class ScenarioRow(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    gpu_name = Column(String, nullable=False)
    dtype = Column(String, nullable=False)
    requests_per_day = Column(Integer, nullable=False)
    peak_multiplier = Column(Float, nullable=False)
    isl = Column(Integer, nullable=False)
    osl = Column(Integer, nullable=False)
    ttft_slo_ms = Column(Float, nullable=False)
    tp = Column(Integer, nullable=False)
    traffic_class = Column(String, nullable=False)
    gpu_mem_util = Column(Float, nullable=False)
    runtime = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)


class CapacityEstimateRow(Base):
    __tablename__ = "capacity_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    replicas = Column(Integer, nullable=False)
    replicas_low = Column(Integer, nullable=False)
    replicas_high = Column(Integer, nullable=False)
    binding_constraint = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    payload = Column(Text, nullable=False)   # JSON blob of derived fields
    created_at = Column(DateTime, nullable=False)


class BenchmarkRunRow(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    tag = Column(String, nullable=False)
    command = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="queued")  # queued/running/done/failed
    stdout = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True)
    result_path = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)


class RecommendationRow(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    summary = Column(Text, nullable=False)   # JSON blob
    confidence = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)


class ReportRow(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    markdown_content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)


def make_engine_and_session(db_url: str) -> Tuple:
    """Create an engine + session factory and ensure all tables exist."""
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, factory
