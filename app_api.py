"""
app_api.py

Minimal but complete FastAPI wrapper around CulvertAnalysisSystem.

Endpoints:
- GET  /health            -> simple health check
- POST /analyze           -> run analysis on-the-fly for a county/state

This version:
- Uses a simple API key header X-API-Key (optional; set via env API_KEY)
- Runs the CulvertAnalysisSystem in a thread so Uvicorn's event loop is not blocked
- Returns JSON with summary and GeoJSON snippets (to avoid huge payloads)
- Keeps everything synchronous inside the thread and is safe for lightweight production use

Place this next to your culvert_analysis.py file.
"""

import os
import json
import traceback
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Import your domain code
from culvert_analysis import CulvertAnalysisSystem

# Configuration
API_KEY = os.getenv("API_KEY", "")  # set to require a key in production, leave empty for dev
MAX_GEOJSON_FEATURES = int(os.getenv("MAX_GEOJSON_FEATURES", "200"))

# FastAPI app
app = FastAPI(title="Culvert Analysis API (Simple)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)

class AnalyzeRequest(BaseModel):
    county: str = Field(..., example="Baltimore")
    state: str = Field(..., example="Maryland")
    country: Optional[str] = Field("USA")
    buffer_distance: Optional[int] = Field(5000, description="Buffer distance in meters")
    include_map: Optional[bool] = Field(False)
    max_culverts: Optional[int] = Field(MAX_GEOJSON_FEATURES)

def require_api_key(x_api_key: Optional[str] = Header(None)):
    """Simple API key dependency. If API_KEY is empty, it allows all (dev)."""
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

def geodf_to_geojson(gdf, max_features=200):
    if gdf is None:
        return None
    try:
        g = gdf.head(max_features).copy() if len(gdf) > max_features else gdf.copy()
        return json.loads(g.to_json())
    except Exception:
        return None

def run_analysis_sync(req_dict):
    """Synchronous worker that runs the full pipeline. Safe to call from a thread."""
    req = AnalyzeRequest(**req_dict)
    cas = CulvertAnalysisSystem()

    # 1. locate
    cas.set_location_by_county_state(req.county, req.state, req.country)
    if cas.bbox is None:
        raise ValueError("Unable to determine bounding box for the requested location. Check county/state names.")

    # 2. collect data
    cas.collect_culvert_data()
    cas.collect_stream_gauge_data()
    cas.collect_flood_event_data()

    # 3. analyses
    cas.proximity_analysis(buffer_distance=req.buffer_distance)
    cas.hydrologic_risk_assessment()
    cas.transportation_impact_analysis()

    # 4. optional model (may skip internally)
    try:
        cas.train_failure_prediction_model()
    except Exception:
        pass

    # 5. scenarios
    try:
        scenarios, synthetic = cas.generate_synthetic_flood_scenarios(n_scenarios=3)
    except Exception:
        scenarios, synthetic = None, None

    # 6. report & serialize outputs
    report = cas.generate_report() or {}

    payload = {
        "report": report,
        "num_culverts_found": len(cas.culverts) if cas.culverts is not None else 0,
        "num_gauges_found": len(cas.gauges) if cas.gauges is not None else 0,
        "num_flood_events_found": len(cas.flood_events) if cas.flood_events is not None else 0,
        "culverts_geojson": geodf_to_geojson(cas.culverts, max_features=req.max_culverts),
        "gauges_geojson": geodf_to_geojson(cas.gauges, max_features=200),
        "flood_events_geojson": geodf_to_geojson(cas.flood_events, max_features=200),
        "scenarios": (scenarios.to_dict(orient="records") if scenarios is not None else None),
        "synthetic_results_sample": (synthetic.head(20).to_dict(orient="records") if synthetic is not None else None),
    }

    if req.include_map:
        try:
            m = cas.create_interactive_map()
            if m is not None:
                payload["map_html"] = m.get_root().render()
        except Exception as e:
            payload["map_html_error"] = str(e)

    return payload

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, _auth=Depends(require_api_key)):
    """
    Main endpoint.
    Runs the analysis in a thread and returns JSON.
    """
    loop = asyncio.get_event_loop()
    try:
        payload = await loop.run_in_executor(executor, partial(run_analysis_sync, req.dict()))
        return payload
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})

# Run with: uvicorn app_api:app --host 0.0.0.0 --port 8000
