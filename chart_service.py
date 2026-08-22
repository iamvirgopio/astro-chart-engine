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


class AstrocartographyRequest(BaseModel):
    julian_day_ut: float  # from the natal chart's own computed output
    query_lat: float = Field(ge=-90, le=90)
    query_lon: float = Field(ge=-180, le=180)
    orb_degrees: float = 6


@app.post("/astrocartography")
def get_astrocartography(req: AstrocartographyRequest):
    """
    Which of the natal chart's planetary lines fall near a given location.
    julian_day_ut comes straight from the natal chart's own /chart output --
    the frontend never recomputes it, just passes it through.
    """
    try:
        lines = ce.compute_astrocartography_lines(req.julian_day_ut)
        hits = ce.check_location_influence(lines, req.query_lat, req.query_lon, req.orb_degrees)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"hits": hits}


class AstrocartographyLinesRequest(BaseModel):
    julian_day_ut: float


@app.post("/astrocartography-lines")
def get_astrocartography_lines(req: AstrocartographyLinesRequest):
    """
    Returns the RAW line geometry (not a proximity check) -- every planet's
    MC/IC longitude and full AC/DC curves. This is what a "recommend me
    places" feature needs: the actual line to sample points along, not a
    yes/no check against one place someone already named.
    """
    try:
        lines = ce.compute_astrocartography_lines(req.julian_day_ut)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"lines": lines}


class SynastryRequest(BaseModel):
    chart_a_positions: dict
    chart_b_positions: dict
    label_a: str = "A"
    label_b: str = "B"


@app.post("/synastry")
def get_synastry(req: SynastryRequest):
    """Every aspect between two charts' planets -- relationship or founder/business comparison."""
    try:
        hits = ce.compute_synastry(req.chart_a_positions, req.chart_b_positions, req.label_a, req.label_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"hits": hits}


class RecommendLocationsRequest(BaseModel):
    julian_day_ut: float
    theme_planets: list[str]
    top_n: int = 5
    orb_degrees: float = 8


@app.post("/recommend-locations")
def get_recommended_locations(req: RecommendLocationsRequest):
    """The real 'where should I go for X' engine -- ranks real candidate
    cities by proximity to the theme's relevant planetary lines."""
    try:
        results = ce.recommend_locations(req.julian_day_ut, req.theme_planets, req.top_n, req.orb_degrees)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results}
