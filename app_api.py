"""
app_api.py

Production-grade FastAPI wrapper around CulvertAnalysisSystem:
- API Key authentication (header: X-API-Key)
- Rate limiting (in-memory fallback + Redis-based limiter)
- Caching via Redis for repeated county/state requests
- Background job support using RQ (enqueue=True to run async and poll via /jobs/{job_id})
- Prometheus metrics endpoint /metrics
- Structured logging
- Optional Postgres persistence for results (via SQLAlchemy)
- Config via environment variables
"""

import os
import json
import time
import asyncio
import hashlib
import logging
import traceback
from functools import partial
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Third-party for production features
import aioredis  # redis >= 2.x not async. We use aioredis which is commonly available.
from redis import Redis
from rq import Queue
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Domain logic
from culvert_analysis import CulvertAnalysisSystem

# -----------------------
# Configuration via env
# -----------------------
API_TITLE = os.getenv("API_TITLE", "Culvert Analysis API")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
API_KEY = os.getenv("API_KEY", "")  # Set a strong key in prod
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # per window
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RQ_REDIS_URL = os.getenv("RQ_REDIS_URL", REDIS_URL)
ENABLE_REDIS = os.getenv("ENABLE_REDIS", "true").lower() == "true"
ENABLE_RQ = os.getenv("ENABLE_RQ", "true").lower() == "true"
ENABLE_DB = os.getenv("ENABLE_DB", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/culverts")
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("app_api")

# -----------------------
# App & Middleware
# -----------------------
app = FastAPI(title=API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Redis/RQ/DB Initialization
# -----------------------
redis_sync: Optional[Redis] = None
redis_async: Optional[Any] = None  # aioredis client
rq_queue: Optional[Queue] = None
db_engine = None

if ENABLE_REDIS:
    try:
        redis_sync = Redis.from_url(REDIS_URL)
        redis_async = asyncio.get_event_loop().run_until_complete(aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True))
        logger.info("Connected to Redis for caching/rate-limit")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Proceeding without Redis.")
        redis_sync = None
        redis_async = None

if ENABLE_RQ and redis_sync is not None:
    try:
        rq_queue = Queue("culvert-jobs", connection=redis_sync)
        logger.info("RQ queue ready")
    except Exception as e:
        logger.warning(f"RQ init failed: {e}")
        rq_queue = None

if ENABLE_DB:
    try:
        db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with db_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id SERIAL PRIMARY KEY,
                    county TEXT NOT NULL,
                    state TEXT NOT NULL,
                    country TEXT NOT NULL,
                    buffer_distance INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    report JSONB,
                    culverts_count INTEGER,
                    gauges_count INTEGER,
                    floods_count INTEGER
                )
            """))
        logger.info("Database connected and table ensured")
    except SQLAlchemyError as e:
        logger.warning(f"DB connection failed: {e}")
        db_engine = None

# -----------------------
# Prometheus metrics
# -----------------------
REQ_COUNTER = Counter("requests_total", "Total requests", ["endpoint", "method", "status"])
REQ_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint", "method"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.time() - start
        endpoint = request.url.path
        method = request.method
        REQ_COUNTER.labels(endpoint=endpoint, method=method, status=status).inc()
        REQ_LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)
    return response

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -----------------------
# Auth dependency
# -----------------------
def api_key_auth(x_api_key: Optional[str] = Header(None)) -> None:
    if not API_KEY:
        # If API_KEY not set, allow all (dev mode). In prod, set API_KEY.
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

# -----------------------
# Rate limiting
# -----------------------
class RateLimiter:
    def __init__(self, requests: int, window_sec: int, redis_client: Optional[Any] = None):
        self.requests = requests
        self.window_sec = window_sec
        self.redis = redis_client
        self.memory_store: Dict[str, Dict[str, Any]] = {}  # fallback

    async def allow(self, key: str) -> bool:
        now = int(time.time())
        window_key = f"ratelimit:{key}:{now // self.window_sec}"

        if self.redis:
            try:
                count = await self.redis.incr(window_key)
                if count == 1:
                    await self.redis.expire(window_key, self.window_sec)
                return count <= self.requests
            except Exception:
                pass

        # Fallback: in-memory simplistic limiter (per-process)
        window_bucket = str(now // self.window_sec)
        entry = self.memory_store.get(key)
        if not entry or entry["bucket"] != window_bucket:
            self.memory_store[key] = {"bucket": window_bucket, "count": 1}
            return True
        else:
            entry["count"] += 1
            return entry["count"] <= self.requests

rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC, redis_async)

async def rate_limit_dep(request: Request):
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}"
    ok = await rate_limiter.allow(key)
    if not ok:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# -----------------------
# Request/Response models
# -----------------------
class AnalyzeRequest(BaseModel):
    county: str = Field(..., example="Baltimore")
    state: str = Field(..., example="Maryland")
    country: Optional[str] = Field("USA", example="USA")
    buffer_distance: Optional[int] = Field(5000, description="Buffer distance in meters for proximity analysis")
    include_map: Optional[bool] = Field(False, description="Whether to include HTML of the interactive map")
    max_culverts: Optional[int] = Field(200, description="Limit culverts in GeoJSON")
    enqueue: Optional[bool] = Field(False, description="If true, run as background job and return job_id")

class AnalyzeResponse(BaseModel):
    job_id: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    num_culverts_found: Optional[int] = 0
    num_gauges_found: Optional[int] = 0
    num_flood_events_found: Optional[int] = 0
    culverts_geojson: Optional[Dict[str, Any]] = None
    gauges_geojson: Optional[Dict[str, Any]] = None
    flood_events_geojson: Optional[Dict[str, Any]] = None
    proximity_pairs_count: Optional[int] = 0
    scenarios: Optional[Any] = None
    synthetic_results_sample: Optional[Any] = None
    ml_model_report: Optional[Dict[str, Any]] = None
    map_html: Optional[str] = None
    map_html_error: Optional[str] = None

# -----------------------
# Utility
# -----------------------
def cache_key_for(req: AnalyzeRequest) -> str:
    key = {
        "county": req.county.strip().lower(),
        "state": req.state.strip().lower(),
        "country": (req.country or "usa").strip().lower(),
        "buffer_distance": req.buffer_distance or 5000,
        "include_map": bool(req.include_map),
        "max_culverts": req.max_culverts or 200,
    }
    raw = json.dumps(key, sort_keys=True)
    return "analysis_cache:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

def serialize_geodf(gdf, max_features=1000):
    if gdf is None:
        return None
    try:
        if hasattr(gdf, "reset_index"):
            g = gdf.head(max_features).copy() if len(gdf) > max_features else gdf.copy()
        else:
            return None
        return json.loads(g.to_json())
    except Exception:
        return None

def persist_summary_to_db(req: AnalyzeRequest, payload: Dict[str, Any]):
    if db_engine is None:
        return
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO analysis_results (county, state, country, buffer_distance, report, culverts_count, gauges_count, floods_count)
                    VALUES (:county, :state, :country, :buffer_distance, :report, :c_count, :g_count, :f_count)
                """),
                {
                    "county": req.county,
                    "state": req.state,
                    "country": req.country or "USA",
                    "buffer_distance": req.buffer_distance or 5000,
                    "report": json.dumps(payload.get("report") or {}),
                    "c_count": payload.get("num_culverts_found", 0),
                    "g_count": payload.get("num_gauges_found", 0),
                    "f_count": payload.get("num_flood_events_found", 0),
                }
            )
    except SQLAlchemyError as e:
        logger.warning(f"DB insert failed: {e}")

def run_full_analysis_sync(req_dict: Dict[str, Any]) -> Dict[str, Any]:
    # This function is RQ-safe (pure sync) and used both in-process and by workers.
    req = AnalyzeRequest(**req_dict)
    cas = CulvertAnalysisSystem()

    # Set location
    cas.set_location_by_county_state(req.county, req.state, req.country)
    if cas.bbox is None:
        raise ValueError("Unable to determine bounding box for the requested location. Check county/state names.")

    # Collect data
    cas.collect_culvert_data()
    cas.collect_stream_gauge_data()
    cas.collect_flood_event_data()

    # Proximity and assessments
    prox = cas.proximity_analysis(buffer_distance=req.buffer_distance)
    cas.hydrologic_risk_assessment()
    cas.transportation_impact_analysis()

    # ML
    model_result = None
    try:
        model_result = cas.train_failure_prediction_model()
    except Exception:
        model_result = None

    # Scenarios
    scenarios, synthetic_results = cas.generate_synthetic_flood_scenarios(n_scenarios=3)

    # Report
    report = cas.generate_report() or {}

    # Serialize output
    payload = {
        "report": report,
        "num_culverts_found": len(cas.culverts) if cas.culverts is not None else 0,
        "num_gauges_found": len(cas.gauges) if cas.gauges is not None else 0,
        "num_flood_events_found": len(cas.flood_events) if cas.flood_events is not None else 0,
        "culverts_geojson": serialize_geodf(cas.culverts, max_features=req.max_culverts or 200),
        "gauges_geojson": serialize_geodf(cas.gauges, max_features=500),
        "flood_events_geojson": serialize_geodf(cas.flood_events, max_features=500),
        "proximity_pairs_count": (len(prox) if prox is not None else 0),
        "scenarios": (scenarios.to_dict(orient="records") if scenarios is not None else None),
        "synthetic_results_sample": (synthetic_results.head(20).to_dict(orient="records") if synthetic_results is not None else None),
        "ml_model_report": (model_result[1] if model_result is not None else None),
    }

    # Map HTML
    if req.include_map:
        try:
            m = cas.create_interactive_map()
            if m is not None:
                payload["map_html"] = m.get_root().render()
        except Exception as e:
            payload["map_html_error"] = str(e)

    return payload

# -----------------------
# Routes
# -----------------------

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    _: None = Depends(api_key_auth),
    __: None = Depends(rate_limit_dep),
):
    # Check cache
    cache_key = cache_key_for(req)
    if redis_sync:
        try:
            cached = redis_sync.get(cache_key)
            if cached:
                logger.info("Cache hit for request")
                return AnalyzeResponse(**json.loads(cached))
        except Exception:
            pass

    # Enqueue as background job if requested
    if req.enqueue:
        if rq_queue is None:
            raise HTTPException(status_code=503, detail="Background queue not available")
        job = rq_queue.enqueue(run_full_analysis_sync, req.dict())
        return AnalyzeResponse(job_id=job.get_id())

    # Otherwise run in-process (sync) but non-blocking to the event loop
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(None, partial(run_full_analysis_sync, req.dict()))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error during analysis: {e}\n{tb}")
        raise HTTPException(status_code=500, detail="Internal error during analysis")

    # Cache response
    if redis_sync:
        try:
            redis_sync.setex(cache_key, CACHE_TTL_SEC, json.dumps(payload))
        except Exception:
            pass

    # Persist summary if DB enabled
    persist_summary_to_db(req, payload)

    return AnalyzeResponse(**payload)

@app.get("/jobs/{job_id}", response_model=AnalyzeResponse)
def get_job(job_id: str, _: None = Depends(api_key_auth)):
    if rq_queue is None:
        raise HTTPException(status_code=503, detail="Background queue not available")
    job = rq_queue.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.is_finished:
        return AnalyzeResponse(job_id=job_id)  # pending
    if job.is_failed:
        raise HTTPException(status_code=500, detail="Job failed")
    result = job.result
    # Cache result too
    if redis_sync and isinstance(result, dict):
        try:
            # Result cache is optional; no deterministic key in this path
            pass
        except Exception:
            pass
    return AnalyzeResponse(**result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app_api:app", host="0.0.0.0", port=port, reload=False, workers=1)
