"""
Chart computation API. Wraps chart_engine.compute_chart() so the Next.js
app can request a chart for ANY user's birth data, not a hardcoded example.

Run locally with: uvicorn chart_service:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import os
import sentry_sdk
import chart_engine as ce
import billing
import push

# Real error monitoring—without this, the only way either of us
# finds out something broke in production is a person telling us,
# which is exactly what happened repeatedly tonight. SENTRY_DSN is set
# in Railway's environment variables; if it's not set (e.g. running
# locally), sentry_sdk.init with an empty dsn is a documented no-op --
# it doesn't error, it just doesn't send anything anywhere.
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN", ""),
    traces_sample_rate=0.1,
    # Full request/response bodies can contain a person's birth data,
    # their typed questions, or session tokens—captured only at the
    # point of an actual unhandled error for debugging it, not
    # continuously logged as a matter of course.
    send_default_pii=False,
)

app = FastAPI(title="Chart Engine Service")
app.include_router(billing.router)
app.include_router(push.router)


def _real_client_ip(request: Request) -> str:
    """Railway (and most reverse proxies) sit in front of this app, so
    request.client.host alone would report the proxy's own address for
    every single request, not the actual visitor—X-Forwarded-For is
    where the real origin IP actually shows up. Falls back to
    request.client.host only if that header is genuinely absent."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_SIGNUP_IP_LIMIT = 3
_SIGNUP_IP_WINDOW_HOURS = 24


@app.post("/signup/check-rate-limit")
def check_signup_rate_limit(request: Request):
    """Called by the frontend BEFORE actually creating a Supabase
    account—rate-limits by IP within a time window rather than a
    lifetime one-email-per-IP block. A hard lifetime block would punish
    every legitimate household, office, or mobile carrier's customers
    who happen to share a public IP with someone who already signed up
    (mobile carriers in particular route huge numbers of genuinely
    different people through a small pool of shared addresses) --
    while barely slowing down anyone actually trying to abuse the
    referral system, since switching from WiFi to cellular data
    defeats a raw IP check in seconds anyway. A short window catches
    the actual abuse pattern (rapid-fire fake signups in one sitting)
    without that cost.
    """
    import os
    from supabase import create_client
    from datetime import datetime, timedelta, timezone

    supabase_admin = create_client(
        os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    )
    ip = _real_client_ip(request)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=_SIGNUP_IP_WINDOW_HOURS)).isoformat()

    recent = supabase_admin.table("signup_ip_log").select("id").eq("ip_address", ip).gte("created_at", cutoff).execute()
    if len(recent.data or []) >= _SIGNUP_IP_LIMIT:
        raise HTTPException(status_code=429, detail="Too many accounts created recently from this connection—try again later.")

    supabase_admin.table("signup_ip_log").insert({"ip_address": ip}).execute()
    return {"allowed": True}


class ChartRequest(BaseModel):
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    # Genuinely optional—an unknown-birth-time chart stores these as
    # null, and compute_chart() already correctly ignores whatever's
    # passed here and defaults to noon whenever unknown_time is True.
    # Requiring a real int here rejected that exact, valid case outright
    # with a 422 before the request ever reached that logic at all.
    hour: int | None = Field(ge=0, le=23, default=12)
    minute: int | None = Field(ge=0, le=59, default=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    unknown_time: bool = False
    chart_system: str = "western"  # "western" (tropical) or "vedic" (sidereal, Lahiri)


@app.post("/chart")
def get_chart(req: ChartRequest):
    try:
        result = ce.compute_chart(
            year=req.year, month=req.month, day=req.day,
            # Falls back to noon if somehow None while unknown_time is
            # False too—an invalid combination that shouldn't occur
            # in practice, but compute_chart() does real arithmetic with
            # these values in that case, unlike the unknown_time=True
            # path, so this guards it defensively rather than assume.
            hour=req.hour if req.hour is not None else 12,
            minute=req.minute if req.minute is not None else 0,
            lat=req.lat, lon=req.lon,
            unknown_time=req.unknown_time,
            chart_system=req.chart_system,
        )
    except Exception as e:
        # Bad coordinates, resolver failures, etc. surface as a clean 400
        # instead of a raw stack trace reaching the frontend.
        raise HTTPException(status_code=400, detail=str(e))
    return result


class SolarReturnRequest(BaseModel):
    natal_chart: dict  # needs .positions.Sun.longitude
    birth_month: int = Field(ge=1, le=12)
    birth_day: int = Field(ge=1, le=31)
    year: int
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ProgressionsRequest(BaseModel):
    natal_chart: dict  # needs .julian_day_ut, the exact birth moment already resolved to UTC
    target_year: int | None = None
    target_month: int | None = None
    target_day: int | None = None
    # Both optional, both required together for progressed angles to be
    # included at all -- see compute_progressed_positions's own
    # docstring for why this needs the real birthplace coordinates
    # rather than being derivable from julian_day_ut alone.
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


@app.post("/progressions")
def get_progressions(req: ProgressionsRequest):
    try:
        birth_jd_ut = req.natal_chart["julian_day_ut"]
        if req.target_year and req.target_month and req.target_day:
            target_jd_ut = ce.julian_day_utc(req.target_year, req.target_month, req.target_day, 12, 0, 0)
        else:
            from datetime import date
            today = date.today()
            target_jd_ut = ce.julian_day_utc(today.year, today.month, today.day, 12, 0, 0)
        result = ce.compute_progressed_positions(birth_jd_ut, target_jd_ut, lat=req.lat, lon=req.lon)
    except KeyError:
        raise HTTPException(status_code=400, detail="natal_chart is missing julian_day_ut—pass the full computed chart, not just positions")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/solar-return")
def get_solar_return(req: SolarReturnRequest):
    """A Solar Return chart—cast for the exact moment the transiting
    Sun returns to the person's natal Sun degree, at wherever they'll
    actually be that year, not their birthplace. Traditionally read as
    its own chart for what that specific year holds, distinct from a
    transit-based year-ahead overview."""
    try:
        natal_sun_lon = req.natal_chart["positions"]["Sun"]["longitude"]
        sr_jd = ce.find_solar_return_jd(natal_sun_lon, req.year, req.birth_month, req.birth_day)
        chart = ce.compute_chart_from_jd_ut(sr_jd, req.lat, req.lon)
        chart["exact_moment_utc"] = ce.jd_to_iso_utc(sr_jd)
        return chart
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    # Optional override—when omitted, behaves exactly as before
    # (today's transits). When provided (e.g. the person's actual
    # birthday, computed client-side from their own stored birth
    # month/day), transits are computed for THAT date instead. This is
    # its own request model rather than reusing VibeOfDayRequest
    # specifically so /vibe-of-day's contract stays untouched and
    # unambiguous—that endpoint must always mean literally today,
    # regardless of what any caller might pass.
    target_year: int | None = None
    target_month: int | None = None
    target_day: int | None = None


@app.post("/today-transits")
def get_today_transits(req: TransitsForDateRequest):
    """Raw, unblended transit content for a given date (why/whats_off,
    no AI layer touching it)—used by the ask page for guidance-style
    questions, which combine this with real natal placement content
    client-side before a single blend call, rather than blending twice.
    Defaults to today; pass target_year/month/day to get transits for
    a specific date instead (e.g. an upcoming birthday)—the actual
    fix for readings that referenced "today" when the real question was
    about a different, known date."""
    from datetime import date
    if req.target_year and req.target_month and req.target_day:
        target = date(req.target_year, req.target_month, req.target_day)
    else:
        target = date.today()
    try:
        # Houses genuinely can't be calculated without a known birth
        # time—houses_and_angles (or this specific house system
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

        # Additive only—reading's own why/whats_off text is untouched,
        # this just also exposes the same top hits as raw structured
        # data (transiting/natal/aspect) rather than already-phrased
        # text, for callers that need to do their own lookup against a
        # hit (e.g. mapping today's transiting planet to a matching
        # crystal) rather than just display generate_reading's prose.
        favorable_hits = [h for h in hits if h["aspect"] in ce.FAVORABLE]
        tense_hits = [h for h in hits if h["aspect"] in ce.TENSE]
        reading["top_favorable_hit"] = max(favorable_hits, key=lambda h: h["weight"]) if favorable_hits else None
        reading["top_tense_hit"] = max(tense_hits, key=lambda h: h["weight"]) if tense_hits else None
        # All hits, not just the single top one—a caller matching
        # hits against a smaller, curated set of content (crystals,
        # for instance, which don't cover every calculated point like
        # Black Moon Lilith or the South Node) needs the option to pick
        # the best-covered hit rather than being stuck with whichever
        # one happened to score highest overall, even if nothing in
        # its own content actually corresponds to it.
        reading["favorable_hits"] = sorted(favorable_hits, key=lambda h: h["weight"], reverse=True)
        reading["tense_hits"] = sorted(tense_hits, key=lambda h: h["weight"], reverse=True)
        # A guaranteed fallback fact—the Moon's current sign needs no
        # aspect to exist at all, unlike every hit above. On a
        # genuinely quiet day (or one where the only real aspects
        # happen to be conjunctions, which count as neither favorable
        # nor tense), this is what keeps a caller like the daily
        # crystal pairing from ever coming up completely empty.
        reading["transiting_moon_sign"] = target_positions["Moon"]["sign"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return reading


class VibeOfDayRequest(BaseModel):
    natal_chart: dict
    house_system: str = "placidus"


_ASPECT_FRAMING = {
    "trine": "a supportive, easy-flowing alignment",
    "sextile": "a supportive alignment that takes some initiative to use well",
    "square": "real friction that pushes growth through tension",
    "opposition": "a pull between two real, competing needs",
    "conjunction": "a blending and intensifying of both energies together",
}


class YearAheadRequest(BaseModel):
    natal_chart: dict
    year: int
    subject_type: str = "personal"  # "personal" | "business"
    subject_label: str | None = None  # the business's name, when subject_type == "business"


NATAL_PLANET_THEME = {
    "Sun": "your core identity and sense of self",
    "Moon": "your emotional needs and instincts",
    "Mercury": "how you think, process, and communicate",
    "Venus": "what you value and how you connect with others",
    "Mars": "your drive, ambition, and how you assert yourself",
    "Jupiter": "how you grow, take risks, and find meaning",
    "Saturn": "your sense of structure, discipline, and long-term responsibility",
    "Uranus": "your need for independence and authenticity",
    "Neptune": "your ideals, intuition, and the parts of yourself that are hardest to pin down",
    "Pluto": "your relationship with power, control, and deep transformation",
    "Chiron": "an old wound you carry, and how you're learning to work with it rather than around it",
    "North Node": "the direction you're meant to be growing toward, even when it's uncomfortable",
}
_NATAL_PLANET_THEME_DEFAULT = "a specific, real part of who you are"

# Mirrors the same convention already used for business charts in Ask
# ("the business's natal X" rather than "your natal X")—the same
# planets, reframed around what each one represents for a business
# specifically rather than a person.
BUSINESS_PLANET_THEME = {
    "Sun": "the business's core identity and brand",
    "Moon": "the business's underlying culture and how it responds to change",
    "Mercury": "its communication, marketing, and day-to-day operations",
    "Venus": "its relationships with clients and partners, and its own brand appeal",
    "Mars": "how assertively it competes and takes action",
    "Jupiter": "where it finds real growth and opportunity",
    "Saturn": "its structure, discipline, and long-term stability",
    "Uranus": "its need to innovate, differentiate, or disrupt",
    "Neptune": "its public image, ideals, and blind spots",
    "Pluto": "power dynamics, competition, and deep transformation within it",
    "Chiron": "a recurring vulnerability the business keeps circling back to",
    "North Node": "the direction the business is meant to be growing toward",
}
_BUSINESS_PLANET_THEME_DEFAULT = "a specific, real part of how the business operates"


@app.post("/year-ahead")
def get_year_ahead(req: YearAheadRequest):
    """The year's real, distinct outer-planet themes, written into one
    genuine overview via the same shared blend function every other
    reading in the app uses—not hand-written content for every
    possible planet combination, which isn't a tractable amount of
    content to write well. Works identically for a business chart --
    the underlying scan doesn't care whose chart it's given, only the
    interpretive framing below changes."""
    try:
        is_business = req.subject_type == "business"
        theme_dict = BUSINESS_PLANET_THEME if is_business else NATAL_PLANET_THEME
        theme_default = _BUSINESS_PLANET_THEME_DEFAULT if is_business else _NATAL_PLANET_THEME_DEFAULT
        subject_noun = f'"{req.subject_label}"' if (is_business and req.subject_label) else ("the business" if is_business else "this person")
        possessive = "its" if is_business else "your"

        hits = ce.scan_year_ahead(req.natal_chart["positions"], req.year, samples=52)

        # Real report: even the two-month-pair version of this still
        # let one month in each pair go completely unmentioned,
        # whichever had the less exact of the two hits (September and
        # December specifically, this time, but it could have been
        # either month in any pair depending on that year's chart).
        # Guaranteeing coverage per PAIR was never actually the goal --
        # coverage per INDIVIDUAL month is, so this is now twelve
        # single-month buckets instead of six paired ones, and the
        # cap that used to force a choice between which months made
        # it in is gone entirely: every month with a real transit gets
        # its tightest-orb hit included, up to all twelve if every
        # month has one. The reading's own length already scales with
        # ingredient count in the shared interpretive voice (see
        # blend_answer's own scaling for why), so a genuinely fuller
        # year of transits produces a longer reading automatically,
        # not a separate thing to configure here.
        buckets = [(m, m) for m in range(1, 13)]
        selected = []
        remaining = list(hits)
        for start_month, end_month in buckets:
            for h in remaining:
                hit_month = int(h["approx_date"][5:7])
                if start_month <= hit_month <= end_month:
                    selected.append(h)
                    remaining.remove(h)
                    break
        top_hits = sorted(selected, key=lambda h: h["approx_date"])

        ingredients = []
        for h in top_hits:
            framing = _ASPECT_FRAMING.get(h["aspect"], "a real alignment")
            natal_theme = theme_dict.get(h["natal"], theme_default)
            transiting_theme = ce.OUTER_PLANET_CROSSING_MEANING.get(h["transiting"], "a real shift")
            month_name = h["approx_date"][:7]
            # Real interpretive content now, not just the mechanical
            # fact—what the transiting planet generally brings,
            # meeting what the natal planet actually represents for
            # this subject, framed by whether the aspect itself is
            # supportive or tense. This is the substance an AI blend
            # can actually turn into meaning; the bare fact alone
            # ("Saturn opposes your natal Sun") gave it nothing to
            # interpret beyond restating the fact itself.
            ingredients.append((
                f"{h['transiting']}_{h['natal']}",
                f"{h['transiting']} forms a {h['aspect']} to {possessive} natal {h['natal']}, closest around {month_name} "
                f"-- {framing}. {h['transiting']} brings {transiting_theme}, meeting {natal_theme}."
            ))
        if not ingredients:
            return {"message": "Nothing especially strong from the outer planets stands out this year—a comparatively quiet one, astrologically."}
        message = ce.blend_answer(
            ingredients,
            f"Explain what {req.year} actually means for {subject_noun} astrologically—not a list of "
            f"transits and dates, but what each real theme below is likely to bring up or require, "
            f"walked through in the order they happen across the year.",
            detailed=True,
            interpretive=True,
        )
        return {"message": message, "themes": top_hits}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/moon-phases")
def get_moon_phases():
    """Current moon phase plus the next 8 upcoming phase transitions with
    their exact dates—the real astronomical data behind the Moon
    Phases page. No personal chart data needed at all, so this takes no
    request body."""
    from datetime import date
    today = date.today()
    jd_now = ce.julian_day_utc(today.year, today.month, today.day, 12, 0, 0)
    current = ce.moon_phase(jd_now)
    upcoming = ce.find_next_moon_phases(jd_now, count=8)
    return {
        "current": current,
        "upcoming": [{"phase": u["phase"], "date": ce.jd_to_iso_utc(u["jd"])} for u in upcoming],
    }


@app.post("/vibe-of-day")
def get_vibe_of_day(req: VibeOfDayRequest):
    """
    The real integrated 'horoscope on steroids'—today's transits,
    retrogrades, eclipse (if any), and moon phase, genuinely blended
    into one cohesive, personalized message via the AI-layer-on-top-of-
    real-content pattern.
    """
    from datetime import date
    today = date.today()
    try:
        # Houses genuinely can't be calculated without a known birth
        # time—houses_and_angles (or this specific house system
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
            today_positions=today_positions, angle_data=house_system_data,
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
    Returns the RAW line geometry (not a proximity check)—every planet's
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
    location—location matters here, this isn't birth-chart data."""
    try:
        result = ce.compute_planetary_hours(req.year, req.month, req.day, req.lat, req.lon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/calendar-range")
def get_calendar_range(req: CalendarRangeRequest):
    """Moon phases, void-of-course windows, and notable transits against
    this specific chart, for a date range—the data layer for the
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
    """Every aspect between two charts' planets—relationship or founder/business comparison."""
    try:
        hits = ce.compute_synastry(req.chart_a_positions, req.chart_b_positions, label_a=req.label_a, label_b=req.label_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"hits": hits}


class ClassifyQuestionRequest(BaseModel):
    question: str
    valid_lenses: list[str]
    context_description: str


class ClassifyQuestionMultiRequest(BaseModel):
    question: str
    valid_lenses: list[str]
    context_description: str
    target_count: int = 3


@app.post("/classify-question-multi")
def classify_question_multi_endpoint(req: ClassifyQuestionMultiRequest):
    """Separate endpoint from /classify-question for the same reason
    the underlying function is separate—Chart & Cards needs several
    relevant houses per question, not one, and this keeps that need
    from ever touching the single-lens path other features depend on."""
    try:
        result = ce.classify_question_multi_lens(
            question_text=req.question, valid_lenses=req.valid_lenses,
            context_description=req.context_description, target_count=req.target_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


class BlendAnswerRequest(BaseModel):
    ingredients: list[list[str]]  # [[label, text], [label, text], ...]
    question: str
    detailed: bool = False
    # Opt-in only—most callers of this shared endpoint (synastry,
    # location questions) have no need for live web search, and it's a
    # real cost/latency/reliability tradeoff not worth defaulting on
    # everywhere. The lookbook is the first real use case: a place
    # mentioned in the occasion text needs genuine, possibly-current
    # context (climate, culture, what people actually wear there) that
    # general model knowledge won't always cover well, especially for
    # less-famous places.
    allow_web_search: bool = False
    # Genuine, explanatory multi-paragraph prose (Year Ahead's voice)
    # rather than the terse concrete-facts default—needed for
    # anything reading a whole chart's worth of placements as one
    # cohesive interpretation, like a Solar Return or Composite chart,
    # rather than answering one specific question.
    interpretive: bool = False
    # See sentence_range_override's docstring entry on blend_answer
    # itself for why this exists -- optional, only used by a caller
    # whose ingredient count doesn't mean what it means for
    # interpretive's own default scaling.
    sentence_range_override: str | None = None
    # See stylist_voice's own docstring entry on _blend_ingredients_into_answer
    # for why this exists as a genuinely separate branch, not a variant
    # of interpretive or the default voice -- currently only used by
    # Star Stylist.
    stylist_voice: bool = False


@app.post("/blend-answer")
def get_blended_answer(req: BlendAnswerRequest):
    """Generic blending endpoint—used by any surface whose real
    content library lives on the frontend (synastry, location, the
    lookbook) rather than in this engine. Takes real ingredients,
    returns one direct, cohesive answer to the actual question asked."""
    try:
        ingredient_tuples = [(item[0], item[1]) for item in req.ingredients]
        message = ce.blend_answer(ingredient_tuples, req.question, detailed=req.detailed, allow_web_search=req.allow_web_search, interpretive=req.interpretive, sentence_range_override=req.sentence_range_override, stylist_voice=req.stylist_voice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": message}


@app.post("/classify-question")
def classify_question_endpoint(req: ClassifyQuestionRequest):
    """General-purpose free-text classifier, reused for synastry and
    location questions—same invisible-AI routing already used for
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
    """The real 'where should I go for X' engine—ranks real candidate
    cities by proximity to the theme's relevant planetary lines."""
    try:
        results = ce.recommend_locations(
            jd_ut=req.julian_day_ut, theme_planets=req.theme_planets,
            theme_lines=req.theme_lines, top_n=req.top_n, orb_degrees=req.orb_degrees,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results}
