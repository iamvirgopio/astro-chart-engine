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
    chart_system: str = "western"  # "western" (tropical) or "vedic" (sidereal, Lahiri)


@app.post("/chart")
def get_chart(req: ChartRequest):
    try:
        result = ce.compute_chart(
            year=req.year, month=req.month, day=req.day,
            hour=req.hour, minute=req.minute,
            lat=req.lat, lon=req.lon,
            unknown_time=req.unknown_time,
            chart_system=req.chart_system,
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


class TransitsForDateRequest(BaseModel):
    natal_chart: dict
    house_system: str = "placidus"
    # Optional override -- when omitted, behaves exactly as before
    # (today's transits). When provided (e.g. the person's actual
    # birthday, computed client-side from their own stored birth
    # month/day), transits are computed for THAT date instead. This is
    # its own request model rather than reusing VibeOfDayRequest
    # specifically so /vibe-of-day's contract stays untouched and
    # unambiguous -- that endpoint must always mean literally today,
    # regardless of what any caller might pass.
    target_year: int | None = None
    target_month: int | None = None
    target_day: int | None = None


@app.post("/today-transits")
def get_today_transits(req: TransitsForDateRequest):
    """Raw, unblended transit content for a given date (why/whats_off,
    no AI layer touching it) -- used by the ask page for guidance-style
    questions, which combine this with real natal placement content
    client-side before a single blend call, rather than blending twice.
    Defaults to today; pass target_year/month/day to get transits for
    a specific date instead (e.g. an upcoming birthday) -- the actual
    fix for readings that referenced "today" when the real question was
    about a different, known date."""
    from datetime import date
    if req.target_year and req.target_month and req.target_day:
        target = date(req.target_year, req.target_month, req.target_day)
    else:
        target = date.today()
    try:
        # Houses genuinely can't be calculated without a known birth
        # time -- houses_and_angles (or this specific house system
        # within it) can legitimately be None for such a chart. The old
        # unconditional subscript here crashed with a bare NoneType
        # error the moment that happened.
        houses_and_angles = req.natal_chart.get("houses_and_angles") or {}
        house_system_data = houses_and_angles.get(req.house_system) or {}
        natal_houses = house_system_data.get("houses")
        jd_ut = ce.julian_day_utc(target.year, target.month, target.day, 12, 0, 0)
        target_positions = ce.compute_positions(jd_ut)
        score, hits = ce.score_day_against_natal(
            target_positions, req.natal_chart["positions"], "timing", natal_houses
        )
        day_result = {"date": target.isoformat(), "score": score, "hits": hits}
        reading = ce.generate_reading(day_result, req.natal_chart["positions"], natal_houses)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return reading


class VibeOfDayRequest(BaseModel):
    natal_chart: dict
    house_system: str = "placidus"


@app.post("/vibe-of-day")
def get_vibe_of_day(req: VibeOfDayRequest):
    """
    The real integrated 'horoscope on steroids' -- today's transits,
    retrogrades, eclipse (if any), and moon phase, genuinely blended
    into one cohesive, personalized message via the AI-layer-on-top-of-
    real-content pattern.
    """
    from datetime import date
    today = date.today()
    try:
        # Houses genuinely can't be calculated without a known birth
        # time -- houses_and_angles (or this specific house system
        # within it) can legitimately be None for such a chart. The old
        # unconditional subscript here crashed with a bare NoneType
        # error the moment that happened.
        houses_and_angles = req.natal_chart.get("houses_and_angles") or {}
        house_system_data = houses_and_angles.get(req.house_system) or {}
        natal_houses = house_system_data.get("houses")
        jd_ut = ce.julian_day_utc(today.year, today.month, today.day, 12, 0, 0)
        today_positions = ce.compute_positions(jd_ut)
        score, hits = ce.score_day_against_natal(
            today_positions, req.natal_chart["positions"], "timing", natal_houses
        )
        day_result = {"date": today.isoformat(), "score": score, "hits": hits}

        retrogrades_today = [
            name for name, data in today_positions.items()
            if name != "_skipped" and name in ce.RETROGRADE_PLANETS and data.get("retrograde")
        ]

        eclipses_today = ce.find_eclipses_in_range(jd_ut - 0.5, jd_ut + 0.5)
        eclipse_today = eclipses_today[0] if eclipses_today else None

        moon_phase_today = ce.moon_phase(jd_ut)

        reading = ce.generate_integrated_vibe_of_day(
            day_result, req.natal_chart["positions"], natal_houses,
            retrogrades_today, eclipse_today, moon_phase_today,
        )
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
        hits = ce.check_location_influence(lines, query_lat=req.query_lat, query_lon=req.query_lon, orb_degrees=req.orb_degrees)
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


class CalendarRangeRequest(BaseModel):
    start_year: int
    start_month: int = Field(ge=1, le=12)
    start_day: int = Field(ge=1, le=31)
    num_days: int = Field(ge=1, le=90, default=31)
    natal_positions: dict | None = None
    natal_houses: list | None = None


class PlanetaryHoursRequest(BaseModel):
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    lat: float
    lon: float


@app.post("/planetary-hours")
def get_planetary_hours(req: PlanetaryHoursRequest):
    """Real sunrise/sunset-based planetary hours for a specific date and
    location -- location matters here, this isn't birth-chart data."""
    try:
        result = ce.compute_planetary_hours(req.year, req.month, req.day, req.lat, req.lon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/calendar-range")
def get_calendar_range(req: CalendarRangeRequest):
    """Moon phases, void-of-course windows, and notable transits against
    this specific chart, for a date range -- the data layer for the
    calendar/planner feature."""
    try:
        result = ce.calendar_range(
            start_year=req.start_year, start_month=req.start_month, start_day=req.start_day,
            num_days=req.num_days, natal_positions=req.natal_positions, natal_houses=req.natal_houses,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/synastry")
def get_synastry(req: SynastryRequest):
    """Every aspect between two charts' planets -- relationship or founder/business comparison."""
    try:
        hits = ce.compute_synastry(req.chart_a_positions, req.chart_b_positions, label_a=req.label_a, label_b=req.label_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"hits": hits}


class ClassifyQuestionRequest(BaseModel):
    question: str
    valid_lenses: list[str]
    context_description: str


class BlendAnswerRequest(BaseModel):
    ingredients: list[list[str]]  # [[label, text], [label, text], ...]
    question: str
    detailed: bool = False


@app.post("/blend-answer")
def get_blended_answer(req: BlendAnswerRequest):
    """Generic blending endpoint -- used by any surface whose real
    content library lives on the frontend (synastry, location, the
    lookbook) rather than in this engine. Takes real ingredients,
    returns one direct, cohesive answer to the actual question asked."""
    try:
        ingredient_tuples = [(item[0], item[1]) for item in req.ingredients]
        message = ce.blend_answer(ingredient_tuples, req.question, detailed=req.detailed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": message}


@app.post("/classify-question")
def classify_question_endpoint(req: ClassifyQuestionRequest):
    """General-purpose free-text classifier, reused for synastry and
    location questions -- same invisible-AI routing already used for
    the main reading flow, just with a swappable lens set."""
    try:
        result = ce.classify_open_question(
            question_text=req.question, valid_lenses=req.valid_lenses,
            context_description=req.context_description,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


class RecommendLocationsRequest(BaseModel):
    julian_day_ut: float
    theme_planets: list[str]
    theme_lines: list[str] | None = None
    top_n: int = 5
    orb_degrees: float = 8


@app.post("/recommend-locations")
def get_recommended_locations(req: RecommendLocationsRequest):
    """The real 'where should I go for X' engine -- ranks real candidate
    cities by proximity to the theme's relevant planetary lines."""
    try:
        results = ce.recommend_locations(
            jd_ut=req.julian_day_ut, theme_planets=req.theme_planets,
            theme_lines=req.theme_lines, top_n=req.top_n, orb_degrees=req.orb_degrees,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results}
