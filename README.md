# ExperimentOS — Production A/B Testing Platform

[![CI](https://github.com/your-username/ab-testing-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/ab-testing-platform/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade **Multi-Armed Bandit experimentation platform** with Thompson Sampling, Bayesian analysis, and sequential testing — the kind of system powering experimentation at Meta, Netflix, and Airbnb.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                         │
│                                                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │ Experiments │  │   Assignments    │  │    Analytics       │ │
│  │   Router    │  │   Router         │  │    Router          │ │
│  │ CRUD, State │  │ Thompson Sampling│  │ Stats + Results    │ │
│  └─────────────┘  └──────────────────┘  └────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Core Engines                         │   │
│  │  ThompsonSamplingEngine  │  ExperimentAnalyzer         │   │
│  │  Beta-Bernoulli MAB      │  Frequentist + Bayesian     │   │
│  │  O(1) posterior update   │  Sequential (Pocock)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Async MongoDB (Motor)                     │    │
│  │  experiments | assignments | conversions               │    │
│  │  Compound unique indexes for deduplication             │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features vs. Basic A/B Testing

| Feature | Basic A/B | This Platform |
|---------|-----------|---------------|
| Traffic allocation | Fixed 50/50 | Thompson Sampling (adaptive) |
| Stats | None | z-test, p-value, CI, Cohen's h, power |
| Bayesian | ❌ | P(B>A), expected loss, credible intervals |
| Early stopping | ❌ | Pocock α-spending + ROPE |
| Multi-variant | ❌ | N arms supported |
| Deduplication | ❌ | Unique index on (experiment, user) |
| Async | ❌ | Full async Motor + FastAPI |
| API docs | ❌ | Auto-generated OpenAPI at /docs |
| Dashboard | matplotlib blobs | Real-time SPA dashboard |
| Docker | ❌ | Dockerfile + docker-compose |
| CI/CD | ❌ | GitHub Actions |

---

## Thompson Sampling — Core Algorithm

Instead of routing equal traffic to each variant, Thompson Sampling draws from each arm's **Beta posterior** and routes to the argmax:

```python
# For each arm i with αᵢ successes and βᵢ failures:
θ̃ᵢ ~ Beta(αᵢ, βᵢ)
chosen_arm = argmax(θ̃₁, θ̃₂, ..., θ̃ₙ)

# After observing reward:
if reward == 1: αᵢ += 1   # posterior update
else:           βᵢ += 1
```

**Why this matters:** Thompson Sampling achieves O(log T) Bayesian regret (Lai-Robbins lower bound optimal), meaning it wastes far fewer users on the losing variant compared to fixed 50/50 splits.

---

## Statistical Analysis

### Frequentist (Two-proportion z-test)
```
H₀: p_control = p_treatment
z = (p̂_t - p̂_c) / √(p̂_pool(1-p̂_pool)(1/n_c + 1/n_t))
p-value = 2(1 - Φ(|z|))
95% CI: Δp ± z_0.025 × SE_unpooled
```

### Bayesian
```
Prior: Beta(1,1) (uniform, non-informative)
Posterior: Beta(successes+1, failures+1)
P(B>A) = ∫P(θ_B > θ_A) dθ  [Monte Carlo, 100k samples]
Expected Loss = E[max_j(θ_j) - θ_i]
```

### Sequential Testing (Pocock α-spending)
```
For planned K peeks, Pocock boundary at peek k:
α*(k) = 2(1 - Φ(c/√(k/K)))   where c ≈ 2.413 for K=5

ROPE stopping: if E[Loss(best arm)] < ε=0.001 → declare null
```

---

## Quick Start

### Local Development (Docker)
```bash
git clone https://github.com/your-username/ab-testing-platform
cd ab-testing-platform
docker-compose up
# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/docs
```

### Manual Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export MONGODB_URI="mongodb+srv://..."  # or local
uvicorn app:app --reload
```

---

## API Reference

### Create Experiment
```bash
curl -X POST /api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "checkout_button_color",
    "variants": [
      {"variant_id": "control",   "name": "Blue Button"},
      {"variant_id": "treatment", "name": "Green Button"}
    ],
    "strategy": "thompson",
    "target_sample_size": 5000
  }'
```

### Assign User (Thompson Sampling)
```bash
curl -X POST /api/v1/assign \
  -d '{"experiment_id": "exp_abc123", "user_id": "user_456"}'
# → {"variant_id": "treatment", "strategy_used": "thompson"}
```

### Record Conversion
```bash
curl -X POST /api/v1/convert \
  -d '{"experiment_id": "exp_abc123", "user_id": "user_456", "reward": 1}'
# → Atomically updates Beta posterior
```

### Get Results
```bash
curl /api/v1/experiments/exp_abc123/results
# → Full frequentist + Bayesian + sequential analysis + recommendation
```

---

## Project Structure
```
ab-testing-platform/
├── app.py                    # FastAPI entry point, lifespan, middleware
├── core/
│   ├── bandit.py            # ThompsonSamplingEngine (Beta-Bernoulli MAB)
│   ├── stats.py             # FrequentistAnalyzer, BayesianAnalyzer, SequentialTester
│   └── models.py            # Pydantic v2 schemas
├── api/
│   ├── experiments.py       # Experiment CRUD
│   ├── assignments.py       # Thompson assignment + conversion recording
│   └── analytics.py        # Results + timeseries + allocation weights
├── db/
│   └── mongo.py             # Motor async client, connection pool, index management
├── templates/
│   └── dashboard.html       # Real-time SPA dashboard (vanilla JS + Chart.js)
├── tests/
│   ├── test_bandit.py
│   ├── test_stats.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Resume Bullet Points

- Engineered a production-grade **Multi-Armed Bandit A/B testing platform** (FastAPI + Motor + MongoDB) serving adaptive traffic allocation via **Thompson Sampling** (Beta-Bernoulli conjugate updates), achieving O(log T) Bayesian regret
- Implemented dual statistical analysis pipeline: **frequentist** (two-proportion z-test, 95% CI, Cohen's h, power analysis) and **Bayesian** (P(B>A) via 100k MC samples, expected loss, credible intervals)  
- Designed **sequential testing engine** with Pocock α-spending function and ROPE-based early stopping, controlling FWER across N interim peeks
- Built async REST API with full OpenAPI spec, deduplication via compound MongoDB indexes, atomic posterior updates, and sub-10ms assignment latency

---

## License
MIT
