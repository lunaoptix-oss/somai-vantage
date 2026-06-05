"""
FastAPI backend for the Multi-Agent Stock Dashboard.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import math
import os


def sanitize(obj):
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj

from orchestrator import analyze_ticker, analyze_tickers_async, DEFAULT_UNIVERSE, feedback

app = FastAPI(title="GS Multi-Agent Stock Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# â”€â”€â”€ Cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_cache: dict = {}
_cache_time: dict = {}
CACHE_TTL = 300  # 5 minutes


def get_cached(ticker: str):
    import time
    if ticker in _cache and time.time() - _cache_time.get(ticker, 0) < CACHE_TTL:
        return _cache[ticker]
    return None


def set_cache(ticker: str, data: dict):
    import time
    _cache[ticker] = data
    _cache_time[ticker] = time.time()


# â”€â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api/analyze/{ticker}")
def analyze_single(ticker: str):
    ticker = ticker.upper()
    cached = get_cached(ticker)
    if cached:
        return cached
    try:
        result = sanitize(analyze_ticker(ticker))
        set_cache(ticker, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ScreenRequest(BaseModel):
    tickers: Optional[List[str]] = None
    top_n: int = 10


@app.post("/api/screen")
async def screen_stocks(req: ScreenRequest):
    tickers = req.tickers or DEFAULT_UNIVERSE
    results = await analyze_tickers_async(tickers)
    # Filter errors
    valid = [r for r in results if "error" not in r]
    # Sort by composite score
    valid.sort(key=lambda x: x.get("summary", {}).get("composite_score", 0), reverse=True)
    return sanitize({"results": valid[:req.top_n], "total_screened": len(tickers)})


@app.get("/api/screen/top")
async def screen_top():
    results = await analyze_tickers_async(DEFAULT_UNIVERSE)
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: x.get("summary", {}).get("composite_score", 0), reverse=True)
    bullish = [r for r in valid if "BULL" in r.get("summary", {}).get("classification", "")]
    bearish = [r for r in valid if "BEAR" in r.get("summary", {}).get("classification", "")]
    return sanitize({
        "top_bullish": bullish[:5],
        "top_bearish": bearish[:5],
        "all": valid,
        "total_screened": len(DEFAULT_UNIVERSE),
    })


@app.get("/api/performance")
def performance():
    summary = feedback.get_performance_summary()
    refinements = feedback.get_agent1_refinements()
    return {"performance": summary, "refinements": refinements}


@app.post("/api/resolve/{signal_id}")
def resolve_signal(signal_id: str, current_price: float):
    result = feedback.resolve_signal(signal_id, current_price)
    if not result:
        raise HTTPException(status_code=404, detail="Signal not found or already resolved")
    return result


@app.get("/api/signals")
def list_signals():
    return {"signals": feedback.history[-50:]}


@app.get("/api/universe")
def get_universe():
    return {"tickers": DEFAULT_UNIVERSE}


if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RAILWAY_ENVIRONMENT") is None
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
