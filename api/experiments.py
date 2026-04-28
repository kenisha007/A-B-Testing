"""
Experiments Router
===================
CRUD for experiment management.
POST   /experiments           → create
GET    /experiments           → list (paginated)
GET    /experiments/{id}      → get one
PATCH  /experiments/{id}/start   → start experiment
PATCH  /experiments/{id}/pause   → pause
PATCH  /experiments/{id}/conclude → conclude
DELETE /experiments/{id}      → delete (soft)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.bandit import ThompsonSamplingEngine
from core.models import (
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
    PaginatedExperiments,
)
from db.mongo import get_db

router = APIRouter(prefix="/experiments", tags=["Experiments"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.post("", response_model=Experiment, status_code=status.HTTP_201_CREATED)
async def create_experiment(payload: ExperimentCreate):
    """Create a new experiment. Initialises Thompson Sampler posteriors."""
    db = get_db()
    exp_id = f"exp_{uuid.uuid4().hex[:12]}"

    # Initialize Thompson Sampler state (Beta(1,1) prior for all arms)
    engine = ThompsonSamplingEngine(
        variant_ids=[v.variant_id for v in payload.variants]
    )
    bandit_state = engine.state_dict()

    doc = {
        "experiment_id": exp_id,
        "status": ExperimentStatus.DRAFT,
        "created_at": _now(),
        "started_at": None,
        "concluded_at": None,
        "bandit_state": bandit_state,
        "peek_count": 0,
        **payload.model_dump(),
    }

    await db.experiments.insert_one(doc)
    return _doc_to_experiment(doc)


@router.get("", response_model=PaginatedExperiments)
async def list_experiments(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    status: Optional[ExperimentStatus] = Query(default=None),
):
    db = get_db()
    query = {}
    if status:
        query["status"] = status

    total = await db.experiments.count_documents(query)
    skip = (page - 1) * per_page

    cursor = db.experiments.find(query).sort("created_at", -1).skip(skip).limit(per_page)
    docs = await cursor.to_list(length=per_page)

    return PaginatedExperiments(
        items=[_doc_to_experiment(d) for d in docs],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{experiment_id}", response_model=Experiment)
async def get_experiment(experiment_id: str):
    doc = await _get_or_404(experiment_id)
    return _doc_to_experiment(doc)


@router.patch("/{experiment_id}/start", response_model=Experiment)
async def start_experiment(experiment_id: str):
    doc = await _get_or_404(experiment_id)
    if doc["status"] not in (ExperimentStatus.DRAFT, ExperimentStatus.PAUSED):
        raise HTTPException(400, "Experiment can only be started from DRAFT or PAUSED state")

    db = get_db()
    await db.experiments.update_one(
        {"experiment_id": experiment_id},
        {"$set": {"status": ExperimentStatus.RUNNING, "started_at": _now()}}
    )
    doc["status"] = ExperimentStatus.RUNNING
    return _doc_to_experiment(doc)


@router.patch("/{experiment_id}/pause", response_model=Experiment)
async def pause_experiment(experiment_id: str):
    doc = await _get_or_404(experiment_id)
    if doc["status"] != ExperimentStatus.RUNNING:
        raise HTTPException(400, "Only RUNNING experiments can be paused")

    db = get_db()
    await db.experiments.update_one(
        {"experiment_id": experiment_id},
        {"$set": {"status": ExperimentStatus.PAUSED}}
    )
    doc["status"] = ExperimentStatus.PAUSED
    return _doc_to_experiment(doc)


@router.patch("/{experiment_id}/conclude", response_model=Experiment)
async def conclude_experiment(experiment_id: str):
    db = get_db()
    await db.experiments.update_one(
        {"experiment_id": experiment_id},
        {"$set": {"status": ExperimentStatus.CONCLUDED, "concluded_at": _now()}}
    )
    doc = await _get_or_404(experiment_id)
    return _doc_to_experiment(doc)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(experiment_id: str):
    await _get_or_404(experiment_id)
    db = get_db()
    # Soft delete: just mark as concluded rather than hard delete
    await db.experiments.update_one(
        {"experiment_id": experiment_id},
        {"$set": {"status": ExperimentStatus.CONCLUDED, "concluded_at": _now(), "_deleted": True}}
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_or_404(experiment_id: str) -> dict:
    db = get_db()
    doc = await db.experiments.find_one(
        {"experiment_id": experiment_id, "_deleted": {"$ne": True}}
    )
    if not doc:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return doc


def _doc_to_experiment(doc: dict) -> Experiment:
    doc = dict(doc)
    doc.pop("_id", None)
    doc.pop("_deleted", None)
    # Ensure variants are dicts
    if "variants" in doc and doc["variants"] and isinstance(doc["variants"][0], dict):
        pass  # already dicts
    return Experiment(**doc)
