"""
Chart computation API. Wraps chart_engine.compute_chart() so the Next.js
app can request a chart for ANY user's birth data, not a hardcoded example.

Run locally with: uvicorn chart_service:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chart_engine as ce

app = FastAPI(title="Chart Engine Service")


class ChartRequest(BaseModel):
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    hour: int = Field(ge=0, le=23, default=12)
    minute: int = Field(ge=0, le=59, default=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    unknown_time: bool = False


@app.post("/chart")
def get_chart(req: ChartRequest):
    try:
        result = ce.compute_chart(
            year=req.year, month=req.month, day=req.day,
            hour=req.hour, minute=req.minute,
            lat=req.lat, lon=req.lon,
            unknown_time=req.unknown_time,
        )
    except Exception as e:
        # Bad coordinates, resolver failures, etc. surface as a clean 400
        # instead of a raw stack trace reaching the frontend.
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


class ReadingRequest(BaseModel):
    question: str
    natal_chart: dict  # the exact JSON that /chart returned at signup, stored in Supabase
    lat: float
    lon: float
    start_year: int
    start_month: int
    start_day: int
    num_days: int = 30


@app.post("/reading")
def get_reading(req: ReadingRequest):
    """
    The actual question-answering endpoint. Takes a free-text question
    plus the user's already-computed natal chart (fetched from Supabase
    on the frontend, not recomputed here), and returns either a full
    reading or a clarify-screen instruction.
    """
    try:
        result = ce.handle_question(
            req.question, req.natal_chart, req.lat, req.lon,
            req.start_year, req.start_month, req.start_day, req.num_days,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


class VibeOfDayRequest(BaseModel):
    natal_chart: dict
    house_system: str = "placidus"


@app.post("/vibe-of-day")
def get_vibe_of_day(req: VibeOfDayRequest):
    """
    Today's single reading, no question needed -- scans just today
    against the natal chart and returns the same verdict/why/whats_off
    structure used everywhere else, so the frontend renders it the same way.
    """
    from datetime import date
    today = date.today()
    try:
        natal_houses = req.natal_chart["houses_and_angles"][req.house_system]["houses"]
        jd_ut = ce.julian_day_utc(today.year, today.month, today.day, 12, 0, 0)
        today_positions = ce.compute_positions(jd_ut)
        score, hits = ce.score_day_against_natal(
            today_positions, req.natal_chart["positions"], "timing", natal_houses
        )
        day_result = {"date": today.isoformat(), "score": score, "hits": hits}
        reading = ce.generate_reading(day_result, req.natal_chart["positions"], natal_houses)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return reading
