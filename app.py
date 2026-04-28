"""
A/B Testing Platform — FastAPI Entry Point
==========================================
Production-grade experimentation platform with:
  - Thompson Sampling (Beta-Bernoulli MAB)
  - Frequentist + Bayesian statistical analysis
  - Sequential testing with Pocock α-spending
  - Async MongoDB (Motor) with proper indexing
  - OpenAPI docs at /docs
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from api.assignments import router as assignments_router
from api.analytics import router as analytics_router
from api.experiments import router as experiments_router
from core.models import HealthResponse
from db.mongo import close_db, init_db, ping_db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting A/B Testing Platform v%s", VERSION)
    await init_db()
    logger.info("✅ Database ready")
    yield
    await close_db()
    logger.info("👋 Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="A/B Testing Platform",
    description="""
## Production-Grade Multi-Armed Bandit Experimentation Platform

### Features
- **Thompson Sampling** — Beta-Bernoulli adaptive traffic allocation
- **Frequentist Analysis** — two-proportion z-test, 95% CI, Cohen's h, power analysis
- **Bayesian Analysis** — P(treatment > control), expected loss, credible intervals
- **Sequential Testing** — Pocock α-spending with ROPE-based early stopping
- **Multi-Variant** — supports A/B/C/n experiments, not just binary splits
- **Idempotent Assignment** — same user always gets the same variant

### Quick Start
1. `POST /experiments` — create an experiment
2. `PATCH /experiments/{id}/start` — start it
3. `POST /assign` — assign users to variants (Thompson Sampling)
4. `POST /convert` — record conversion events
5. `GET /experiments/{id}/results` — view full statistical analysis
    """,
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Log request duration for latency monitoring."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    if duration_ms > 500:
        logger.warning("Slow request: %s %s took %.1fms", request.method, request.url.path, duration_ms)
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(experiments_router, prefix="/api/v1")
app.include_router(assignments_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    db_ok = await ping_db()
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=VERSION,
        db_connected=db_ok,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def dashboard():
    return FileResponse("templates/dashboard.html")


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "dev") == "dev",
        log_level="info",
    )
