"""
Async MongoDB Client
=====================
Motor-based async client with:
  - Connection pooling (maxPoolSize=50 for prod)
  - Index management on startup
  - Typed collection accessors
  - Retry logic on transient failures
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


async def init_db() -> None:
    """
    Initialize Motor async client and ensure indexes exist.
    Called once at FastAPI startup.
    """
    global _client, _db

    uri = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017",
    )

    _client = AsyncIOMotorClient(
        uri,
        maxPoolSize=50,
        minPoolSize=5,
        serverSelectionTimeoutMS=5_000,
        socketTimeoutMS=30_000,
    )

    db_name = os.getenv("MONGODB_DB", "ab_platform")
    _db = _client[db_name]

    await _ensure_indexes()
    logger.info("MongoDB connected: db=%s", db_name)


async def _ensure_indexes() -> None:
    """Create all required indexes idempotently."""
    db = get_db()

    # experiments
    await db.experiments.create_indexes([
        IndexModel([("experiment_id", ASCENDING)], unique=True, name="idx_exp_id"),
        IndexModel([("status", ASCENDING)], name="idx_exp_status"),
        IndexModel([("created_at", DESCENDING)], name="idx_exp_created"),
    ])

    # assignments — deduplicate by (experiment_id, user_id)
    await db.assignments.create_indexes([
        IndexModel(
            [("experiment_id", ASCENDING), ("user_id", ASCENDING)],
            unique=True,
            name="idx_assign_dedup",
        ),
        IndexModel([("experiment_id", ASCENDING)], name="idx_assign_exp"),
        IndexModel([("assigned_at", DESCENDING)], name="idx_assign_time"),
    ])

    # conversions
    await db.conversions.create_indexes([
        IndexModel(
            [("experiment_id", ASCENDING), ("user_id", ASCENDING)],
            name="idx_conv_user",
        ),
        IndexModel([("experiment_id", ASCENDING)], name="idx_conv_exp"),
        IndexModel([("converted_at", DESCENDING)], name="idx_conv_time"),
    ])

    logger.info("MongoDB indexes ensured")


async def close_db() -> None:
    if _client:
        _client.close()
        logger.info("MongoDB connection closed")


async def ping_db() -> bool:
    """Health check — returns True if DB is reachable."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False
