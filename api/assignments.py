"""
Assignments & Conversions Router
==================================
POST /assign   → Thompson Sampling variant assignment (idempotent per user)
POST /convert  → Record a conversion event and update Bayesian posterior
GET  /assignments/{experiment_id}/user/{user_id} → check existing assignment
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from core.bandit import ThompsonSamplingEngine
from core.models import (
    AssignmentRequest,
    AssignmentResponse,
    ConversionEvent,
    ExperimentStatus,
    TrafficAllocationStrategy,
)
from db.mongo import get_db

router = APIRouter(tags=["Traffic & Conversions"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/assign", response_model=AssignmentResponse)
async def assign_user(payload: AssignmentRequest):
    """
    Assign a user to a variant using the experiment's configured strategy.

    Idempotent: if the user already has an assignment for this experiment,
    the same variant is returned (no new assignment created).

    Thompson Sampling: draws θ̃ ~ Beta(α,β) for each arm and routes to argmax.
    """
    db = get_db()

    # Load experiment
    exp_doc = await db.experiments.find_one({"experiment_id": payload.experiment_id})
    if not exp_doc:
        raise HTTPException(404, f"Experiment '{payload.experiment_id}' not found")
    if exp_doc["status"] != ExperimentStatus.RUNNING:
        raise HTTPException(400, f"Experiment is {exp_doc['status']}, not RUNNING")

    # Check for existing assignment (idempotent)
    existing = await db.assignments.find_one({
        "experiment_id": payload.experiment_id,
        "user_id": payload.user_id,
    })
    if existing:
        return AssignmentResponse(
            experiment_id=payload.experiment_id,
            user_id=payload.user_id,
            variant_id=existing["variant_id"],
            assignment_id=existing["assignment_id"],
            strategy_used=exp_doc.get("strategy", TrafficAllocationStrategy.THOMPSON),
            assigned_at=existing["assigned_at"],
        )

    # Select variant
    strategy = exp_doc.get("strategy", TrafficAllocationStrategy.THOMPSON)
    variant_ids = [v["variant_id"] for v in exp_doc["variants"]]

    if strategy == TrafficAllocationStrategy.THOMPSON:
        engine = ThompsonSamplingEngine.from_state_dict(exp_doc["bandit_state"])
        variant_id = engine.select_arm()

    elif strategy == TrafficAllocationStrategy.EPSILON_GREEDY:
        import random, numpy as np
        epsilon = 0.1
        state = exp_doc["bandit_state"]
        means = {vid: state[vid]["alpha"] / (state[vid]["alpha"] + state[vid]["beta"])
                 for vid in variant_ids}
        if random.random() < epsilon:
            variant_id = random.choice(variant_ids)
        else:
            variant_id = max(means, key=means.__getitem__)

    else:  # EQUAL
        import random
        variant_id = random.choice(variant_ids)

    assignment_id = f"asgn_{uuid.uuid4().hex[:16]}"
    assignment_doc = {
        "assignment_id": assignment_id,
        "experiment_id": payload.experiment_id,
        "user_id": payload.user_id,
        "variant_id": variant_id,
        "strategy_used": strategy,
        "context": payload.context,
        "assigned_at": _now(),
    }

    # Upsert to handle race conditions (unique index on experiment_id + user_id)
    try:
        await db.assignments.insert_one(assignment_doc)
    except Exception:
        # Race condition: another request assigned this user simultaneously
        existing = await db.assignments.find_one({
            "experiment_id": payload.experiment_id,
            "user_id": payload.user_id,
        })
        if existing:
            return AssignmentResponse(
                experiment_id=payload.experiment_id,
                user_id=payload.user_id,
                variant_id=existing["variant_id"],
                assignment_id=existing["assignment_id"],
                strategy_used=strategy,
                assigned_at=existing["assigned_at"],
            )
        raise

    return AssignmentResponse(
        experiment_id=payload.experiment_id,
        user_id=payload.user_id,
        variant_id=variant_id,
        assignment_id=assignment_id,
        strategy_used=strategy,
        assigned_at=assignment_doc["assigned_at"],
    )


@router.post("/convert", status_code=status.HTTP_200_OK)
async def record_conversion(payload: ConversionEvent):
    """
    Record a conversion event.

    Atomically:
      1. Validates the user was assigned to this experiment
      2. Stores the conversion event
      3. Updates the Thompson Sampler posterior in-place (Beta conjugate update)

    Deduplication: one conversion per (experiment_id, user_id) — later events are ignored.
    """
    db = get_db()

    # Verify assignment exists
    assignment = await db.assignments.find_one({
        "experiment_id": payload.experiment_id,
        "user_id": payload.user_id,
    })
    if not assignment:
        raise HTTPException(
            400,
            f"User '{payload.user_id}' has no assignment for experiment '{payload.experiment_id}'. "
            "Call /assign first."
        )

    variant_id = assignment["variant_id"]

    # Deduplication: check if conversion already recorded
    already_converted = await db.conversions.find_one({
        "experiment_id": payload.experiment_id,
        "user_id": payload.user_id,
    })
    if already_converted:
        return {"status": "duplicate_ignored", "variant_id": variant_id}

    # Record conversion
    conv_doc = {
        "conversion_id": f"conv_{uuid.uuid4().hex[:16]}",
        "experiment_id": payload.experiment_id,
        "user_id": payload.user_id,
        "variant_id": variant_id,
        "reward": payload.reward,
        "event_type": payload.event_type,
        "metadata": payload.metadata,
        "converted_at": _now(),
    }
    await db.conversions.insert_one(conv_doc)

    # Atomic posterior update on experiment document
    binary_reward = 1 if payload.reward >= 0.5 else 0
    if binary_reward == 1:
        await db.experiments.update_one(
            {"experiment_id": payload.experiment_id},
            {"$inc": {f"bandit_state.{variant_id}.alpha": 1.0}}
        )
    else:
        await db.experiments.update_one(
            {"experiment_id": payload.experiment_id},
            {"$inc": {f"bandit_state.{variant_id}.beta": 1.0}}
        )

    return {"status": "recorded", "variant_id": variant_id, "reward": payload.reward}


@router.get("/assignments/{experiment_id}/user/{user_id}")
async def get_user_assignment(experiment_id: str, user_id: str):
    """Check the current variant assignment for a specific user."""
    db = get_db()
    doc = await db.assignments.find_one(
        {"experiment_id": experiment_id, "user_id": user_id},
        {"_id": 0}
    )
    if not doc:
        raise HTTPException(404, "No assignment found")
    return doc
