"""
Analytics Router
=================
GET /experiments/{id}/results   → Full frequentist + Bayesian + sequential analysis
GET /experiments/{id}/timeseries → Conversion rate over time (for dashboard charts)
GET /experiments/{id}/allocation → Current Thompson allocation weights
POST /experiments/{id}/peek      → Trigger an interim analysis (sequential testing)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from core.bandit import ThompsonSamplingEngine
from core.models import (
    BayesianSummary,
    ExperimentResults,
    ExperimentStatus,
    FrequentistSummary,
    SequentialSummary,
    VariantSummary,
)
from core.stats import ExperimentAnalyzer, VariantStats
from db.mongo import get_db

router = APIRouter(tags=["Analytics"])


@router.get("/experiments/{experiment_id}/results", response_model=ExperimentResults)
async def get_results(experiment_id: str):
    """
    Full experiment analysis. For A/B (2-variant) experiments returns:
      - Per-variant: trials, conversions, rate, CI, posterior, P(best)
      - Frequentist: z-stat, p-value, lift, CI, Cohen's h, power
      - Bayesian: P(treatment > control), expected loss, credible intervals
      - Sequential: Pocock boundary check, ROPE decision, early-stop recommendation
      - Holistic recommendation: ship / continue / stop_null
    """
    db = get_db()
    exp_doc = await db.experiments.find_one({"experiment_id": experiment_id})
    if not exp_doc:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")

    # Aggregate assignment + conversion counts per variant
    variant_ids = [v["variant_id"] for v in exp_doc["variants"]]
    variant_names = {v["variant_id"]: v["name"] for v in exp_doc["variants"]}

    # Assignment counts
    assignment_pipeline = [
        {"$match": {"experiment_id": experiment_id}},
        {"$group": {"_id": "$variant_id", "count": {"$sum": 1}}}
    ]
    conv_pipeline = [
        {"$match": {"experiment_id": experiment_id}},
        {"$group": {"_id": "$variant_id", "count": {"$sum": 1}}}
    ]

    assignment_cursor = db.assignments.aggregate(assignment_pipeline)
    conversion_cursor = db.conversions.aggregate(conv_pipeline)

    assignments_map: Dict[str, int] = {}
    async for doc in assignment_cursor:
        assignments_map[doc["_id"]] = doc["count"]

    conversions_map: Dict[str, int] = {}
    async for doc in conversion_cursor:
        conversions_map[doc["_id"]] = doc["count"]

    # Build per-variant stats
    bandit_state = exp_doc.get("bandit_state", {})
    engine = ThompsonSamplingEngine.from_state_dict(bandit_state) if bandit_state else None
    prob_best = engine.compute_prob_best() if engine else {vid: 1.0 / len(variant_ids) for vid in variant_ids}

    variant_summaries: List[VariantSummary] = []
    stats_map: Dict[str, VariantStats] = {}

    for vid in variant_ids:
        trials = assignments_map.get(vid, 0)
        convs  = conversions_map.get(vid, 0)
        rate   = convs / trials if trials > 0 else 0.0
        alpha_p = bandit_state.get(vid, {}).get("alpha", 1.0)
        beta_p  = bandit_state.get(vid, {}).get("beta", 1.0)

        from scipy.stats import beta as beta_dist
        ci_lo = float(beta_dist.ppf(0.025, alpha_p, beta_p))
        ci_hi = float(beta_dist.ppf(0.975, alpha_p, beta_p))

        variant_summaries.append(VariantSummary(
            variant_id=vid,
            name=variant_names.get(vid, vid),
            trials=trials,
            conversions=convs,
            conversion_rate=round(rate, 6),
            conversion_rate_ci_lower=round(ci_lo, 6),
            conversion_rate_ci_upper=round(ci_hi, 6),
            posterior_alpha=alpha_p,
            posterior_beta=beta_p,
            prob_best=round(prob_best.get(vid, 0.0), 4),
        ))
        stats_map[vid] = VariantStats(vid, trials, convs)

    total_assignments = sum(assignments_map.values())
    total_conversions  = sum(conversions_map.values())

    # Statistical analysis (only meaningful for exactly 2 variants for now)
    freq_summary = None
    bayes_summary = None
    seq_summary   = None
    recommendation = "needs_more_data"
    confidence     = "low"

    if len(variant_ids) == 2:
        control_id   = variant_ids[0]
        treatment_id = variant_ids[1]
        control_stats   = stats_map[control_id]
        treatment_stats = stats_map[treatment_id]

        analyzer = ExperimentAnalyzer(
            alpha=exp_doc.get("alpha", 0.05),
            total_peeks=exp_doc.get("total_peeks", 5),
        )
        analysis = analyzer.analyze(
            control_stats, treatment_stats,
            peek_number=max(exp_doc.get("peek_count", 1), 1)
        )

        f = analysis.frequentist
        freq_summary = FrequentistSummary(
            p_value=round(f.p_value, 6),
            z_statistic=round(f.z_stat, 4),
            absolute_lift=round(f.absolute_lift, 6),
            relative_lift=round(f.relative_lift, 4),
            ci_lower=round(f.ci_lower, 6),
            ci_upper=round(f.ci_upper, 6),
            cohens_h=round(f.cohens_h, 4),
            is_significant=f.is_significant,
            observed_power=round(f.power, 4),
            min_detectable_effect=round(f.min_detectable_effect, 6),
        )

        b = analysis.bayesian
        bayes_summary = BayesianSummary(
            prob_treatment_better=round(b.prob_treatment_better, 4),
            expected_loss_control=round(b.expected_loss_control, 6),
            expected_loss_treatment=round(b.expected_loss_treatment, 6),
            ci_control=list(map(lambda x: round(x, 4), b.credible_interval_control)),
            ci_treatment=list(map(lambda x: round(x, 4), b.credible_interval_treatment)),
            bayes_factor=round(min(b.bayes_factor, 9999.0), 2),
        )

        s = analysis.sequential
        seq_summary = SequentialSummary(
            peek_number=s.peek_number,
            total_peeks_planned=s.total_peeks_planned,
            alpha_boundary=round(s.alpha_boundary, 6),
            current_p_value=round(f.p_value, 6),
            can_stop_early=s.can_stop,
            rope_decision=s.rope_decision,
        )

        recommendation = analysis.recommendation
        confidence     = analysis.confidence_level

    return ExperimentResults(
        experiment_id=experiment_id,
        experiment_name=exp_doc["name"],
        status=exp_doc["status"],
        variants=variant_summaries,
        frequentist=freq_summary,
        bayesian=bayes_summary,
        sequential=seq_summary,
        recommendation=recommendation,
        confidence_level=confidence,
        total_assignments=total_assignments,
        total_conversions=total_conversions,
        computed_at=datetime.now(timezone.utc),
    )


@router.get("/experiments/{experiment_id}/timeseries")
async def get_timeseries(experiment_id: str, bucket: str = "day"):
    """
    Conversion rates over time, bucketed by hour/day/week.
    Used to power the time-series chart in the dashboard.
    """
    db = get_db()
    exp_doc = await db.experiments.find_one({"experiment_id": experiment_id})
    if not exp_doc:
        raise HTTPException(404, "Experiment not found")

    bucket_formats = {"hour": "%Y-%m-%dT%H", "day": "%Y-%m-%d", "week": "%Y-W%V"}
    date_format = bucket_formats.get(bucket, "%Y-%m-%d")

    pipeline = [
        {"$match": {"experiment_id": experiment_id}},
        {"$group": {
            "_id": {
                "bucket": {"$dateToString": {"format": date_format, "date": "$converted_at"}},
                "variant": "$variant_id",
            },
            "conversions": {"$sum": 1},
        }},
        {"$sort": {"_id.bucket": 1}},
    ]

    results: List[Dict[str, Any]] = []
    async for doc in db.conversions.aggregate(pipeline):
        results.append({
            "bucket": doc["_id"]["bucket"],
            "variant": doc["_id"]["variant"],
            "conversions": doc["conversions"],
        })

    return {"experiment_id": experiment_id, "bucket": bucket, "data": results}


@router.get("/experiments/{experiment_id}/allocation")
async def get_allocation_weights(experiment_id: str):
    """
    Current Thompson Sampling allocation weights based on posterior means.
    Shows how traffic is being dynamically routed toward the winning arm.
    """
    db = get_db()
    exp_doc = await db.experiments.find_one({"experiment_id": experiment_id})
    if not exp_doc:
        raise HTTPException(404, "Experiment not found")

    bandit_state = exp_doc.get("bandit_state", {})
    if not bandit_state:
        return {"weights": {}}

    engine = ThompsonSamplingEngine.from_state_dict(bandit_state)
    weights = engine.allocation_weights()
    prob_best = engine.compute_prob_best()

    return {
        "experiment_id": experiment_id,
        "strategy": exp_doc.get("strategy"),
        "allocation_weights": {vid: round(w, 4) for vid, w in weights.items()},
        "prob_best": {vid: round(p, 4) for vid, p in prob_best.items()},
        "posterior_means": {
            vid: round(arm.mean, 4) for vid, arm in engine.arms.items()
        },
    }


@router.post("/experiments/{experiment_id}/peek")
async def trigger_peek(experiment_id: str):
    """
    Trigger an interim analysis (sequential testing peek).
    Increments peek_count and re-evaluates the Pocock stopping boundary.
    """
    db = get_db()
    result = await db.experiments.update_one(
        {"experiment_id": experiment_id},
        {"$inc": {"peek_count": 1}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Experiment not found")

    exp_doc = await db.experiments.find_one({"experiment_id": experiment_id})
    return {
        "experiment_id": experiment_id,
        "peek_count": exp_doc["peek_count"],
        "message": "Peek recorded. Call /results to see updated sequential analysis."
    }
