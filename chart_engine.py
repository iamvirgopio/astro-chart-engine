"""
Core chart computation engine.
Takes birth (or founding) date/time/location, returns planetary positions,
house cusps (Placidus + Whole Sign), and angles.

Uses Moshier semi-analytic mode (no external ephemeris data files needed,
~1 arcsecond precision — plenty for astrology). Swap to swe.set_ephe_path()
+ downloaded .se1 files later if sub-arcsecond precision is ever needed.
"""

import swisseph as swe
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder

_tf = TimezoneFinder()  # loaded once; reuse across calls

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY,
    "Venus": swe.VENUS, "Mars": swe.MARS, "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN, "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO, "Chiron": swe.CHIRON,
    "North Node": swe.TRUE_NODE,
}

FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED  # no data files required


def deg_to_sign(longitude):
    """Convert absolute ecliptic longitude (0-360) to sign + degree-within-sign."""
    sign_index = int(longitude // 30)
    degree_in_sign = longitude % 30
    return SIGNS[sign_index], round(degree_in_sign, 2)


def resolve_utc_offset(year, month, day, hour, minute, lat, lon):
    """
    Given local birth date/time + coordinates, returns the correct
    HISTORICAL UTC offset in hours for that exact moment — accounting for
    DST rules as they actually stood on that date, not today's rules.
    Uses the IANA tz database (via zoneinfo), which tracks historical
    DST changes per region — this is the piece that makes "just tell me
    the city" enough, without the user needing to know their own offset.
    """
    tz_name = _tf.timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        raise ValueError(
            f"Could not resolve a timezone for coordinates ({lat}, {lon}). "
            "Likely open ocean or bad geocoding — validate location input upstream."
        )
    local_dt = datetime(year, month, day, hour, minute, tzinfo=ZoneInfo(tz_name))
    offset = local_dt.utcoffset()
    return offset.total_seconds() / 3600.0, tz_name


def julian_day_utc(year, month, day, hour, minute, utc_offset_hours):
    """
    hour/minute are LOCAL time at birth; utc_offset_hours is what to subtract
    to get UTC (e.g. US Central Daylight = -5, so UTC = local - (-5) = local + 5).
    Caller is responsible for resolving the correct historical UTC offset
    for the date/location (DST rules change over time).
    """
    decimal_hour_local = hour + minute / 60.0
    decimal_hour_utc = decimal_hour_local - utc_offset_hours
    return swe.julday(year, month, day, decimal_hour_utc)


def compute_positions(jd_ut):
    """
    Raw planetary positions at a given Julian Day (UT).
    Chiron requires a downloaded asteroid seed file (seas_18.se1) even in
    Moshier mode — if it's not present, we skip Chiron rather than fail
    the whole chart. Flagged in the result so the caller knows it's missing.
    """
    positions = {}
    skipped = []
    for name, code in PLANETS.items():
        try:
            xx, _ = swe.calc_ut(jd_ut, code, FLAGS)
        except swe.Error:
            skipped.append(name)
            continue
        lon = xx[0]
        speed = xx[3]
        sign, deg = deg_to_sign(lon)
        positions[name] = {
            "longitude": round(lon, 4),
            "sign": sign,
            "degree_in_sign": deg,
            "retrograde": speed < 0,
            "speed_deg_per_day": round(speed, 4),
        }
    if skipped:
        positions["_skipped"] = skipped
    return positions


def compute_angles_and_houses(jd_ut, lat, lon):
    """Returns Placidus houses/angles and Whole Sign houses/angles from one calc."""
    # Placidus
    cusps_p, ascmc_p = swe.houses(jd_ut, lat, lon, b'P')
    asc_p, mc_p = ascmc_p[0], ascmc_p[1]
    asc_sign, asc_deg = deg_to_sign(asc_p)
    mc_sign, mc_deg = deg_to_sign(mc_p)

    placidus = {
        "ascendant": {"sign": asc_sign, "degree_in_sign": asc_deg, "longitude": round(asc_p, 4)},
        "midheaven": {"sign": mc_sign, "degree_in_sign": mc_deg, "longitude": round(mc_p, 4)},
        "houses": [
            {"house": i + 1, **dict(zip(["sign", "degree_in_sign"], deg_to_sign(cusps_p[i])))}
            for i in range(12)
        ],
    }

    # Whole Sign: house 1 = the Ascendant's whole sign; each subsequent house
    # is simply the next sign in order, cusp at 0 degrees of that sign.
    asc_sign_index = SIGNS.index(asc_sign)
    whole_sign = {
        "ascendant": placidus["ascendant"],  # Asc degree is the same point regardless of house system
        "midheaven": placidus["midheaven"],
        "houses": [
            {"house": i + 1, "sign": SIGNS[(asc_sign_index + i) % 12], "degree_in_sign": 0.0}
            for i in range(12)
        ],
    }

    return {"placidus": placidus, "whole_sign": whole_sign}


ASPECTS = {
    "conjunction": (0, 8),
    "sextile": (60, 6),
    "square": (90, 8),
    "trine": (120, 8),
    "opposition": (180, 8),
}
# (angle, orb) -- orb is how many degrees off-exact still counts as the aspect

FAVORABLE = {"sextile", "trine"}
TENSE = {"square", "opposition"}
NEUTRAL = {"conjunction"}  # conjunction's tone depends on which planets -- context-dependent

# How much a TRANSITING planet's aspect should count toward a day's score.
# Personal/fast planets are the actual timing signal -- they move degrees per
# day, so "in aspect today, gone tomorrow" is meaningful. Outer planets move
# so slowly that an aspect can sit "in orb" for weeks or months straight;
# left unweighted they drown out every real signal with constant noise.
# This profile is tuned for short-range TIMING questions (launch dates,
# good days for X); a "life themes" lens would flip these weights around.
TIMING_WEIGHTS = {
    "Sun": 1.0, "Moon": 1.2, "Mercury": 0.9, "Venus": 0.9, "Mars": 1.0,
    "Jupiter": 0.4, "Saturn": 0.3,
    "Uranus": 0.08, "Neptune": 0.08, "Pluto": 0.08, "North Node": 0.15,
}


def angular_distance(lon1, lon2):
    """Shortest angular distance between two ecliptic longitudes, 0-180."""
    d = abs(lon1 - lon2) % 360
    return min(d, 360 - d)


def find_aspect(lon1, lon2):
    """Returns (aspect_name, exactness_in_degrees) if lon1/lon2 form a
    recognized aspect within orb, else None. Exactness = how close to
    perfect (0 = exact)."""
    dist = angular_distance(lon1, lon2)
    for name, (angle, orb) in ASPECTS.items():
        diff = abs(dist - angle)
        if diff <= orb:
            return name, round(diff, 2)
    return None


def which_house(longitude, houses):
    """
    Given an absolute ecliptic longitude and a house list (from
    compute_angles_and_houses -- either the placidus or whole_sign list),
    returns which house number (1-12) that longitude falls in.
    """
    SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    cusps = []
    for h in houses:
        sidx = SIGNS.index(h["sign"])
        cusps.append(sidx * 30 + h["degree_in_sign"])
    for i in range(12):
        start = cusps[i]
        end = cusps[(i + 1) % 12]
        span = (end - start) % 360
        offset = (longitude - start) % 360
        if offset < span:
            return i + 1
    return 12  # fallback, shouldn't normally hit


# Each lens: which houses matter for that kind of question, and which
# planets get an extra boost when they're the ones making the aspect.
# House bonus applies when the NATAL planet being aspected sits in one
# of the lens's relevant houses -- e.g. for a money question, a transit
# hitting your natal planet that happens to live in your 2nd or 8th
# house matters more than the same transit hitting a planet in your 5th.
QUESTION_LENSES = {
    "timing": {
        "houses": [], "house_bonus": 0,
        "planet_weights": TIMING_WEIGHTS,
        "description": "General favorable/unfavorable timing (default)",
    },
    "money": {
        "houses": [2, 8], "house_bonus": 0.5,
        "planet_weights": {**TIMING_WEIGHTS, "Venus": 1.3, "Jupiter": 0.7},
        "description": "Earning, income, investments, shared resources",
    },
    "career": {
        "houses": [6, 10], "house_bonus": 0.5,
        "planet_weights": {**TIMING_WEIGHTS, "Saturn": 0.6, "Sun": 1.2},
        "description": "Public standing, work, business launches",
    },
    "relationships": {
        "houses": [7], "house_bonus": 0.5,
        "planet_weights": {**TIMING_WEIGHTS, "Venus": 1.3, "Moon": 1.3},
        "description": "Partnership, synastry-adjacent solo timing",
    },
}


def score_day_against_natal(transiting_positions, natal_positions,
                             lens="timing", natal_houses=None):
    """
    Compares one day's transiting positions against a natal chart,
    scored through a QUESTION_LENSES profile. natal_houses (a placidus
    or whole_sign house list) is required for any lens with a house
    bonus -- without it, the house bonus is silently skipped, which
    degrades to plain timing-weighted scoring.
    """
    profile = QUESTION_LENSES.get(lens, QUESTION_LENSES["timing"])
    weights = profile["planet_weights"]
    relevant_houses = set(profile["houses"])
    house_bonus = profile["house_bonus"]

    hits = []
    score = 0
    for t_name, t_data in transiting_positions.items():
        if t_name == "_skipped":
            continue
        planet_weight = weights.get(t_name, 0.5)
        for n_name, n_data in natal_positions.items():
            if n_name == "_skipped":
                continue
            result = find_aspect(t_data["longitude"], n_data["longitude"])
            if result:
                aspect_name, exactness = result
                orb_weight = 1 - (exactness / 10)
                weight = planet_weight * orb_weight

                bonus_applied = False
                if natal_houses and relevant_houses:
                    house_num = which_house(n_data["longitude"], natal_houses)
                    if house_num in relevant_houses:
                        weight += house_bonus
                        bonus_applied = True

                if aspect_name in FAVORABLE:
                    score += weight
                elif aspect_name in TENSE:
                    score -= weight
                hits.append({
                    "transiting": t_name, "natal": n_name,
                    "aspect": aspect_name, "orb": exactness,
                    "weight": round(weight, 3), "house_bonus": bonus_applied,
                })
    return round(score, 2), hits


def scan_date_range(natal_positions, start_year, start_month, start_day,
                     num_days, lat, lon, lens="timing", natal_houses=None):
    """
    Scores each day in a range against the natal chart through the
    given question lens. Uses noon UTC for each candidate day (transits
    move slowly enough day-to-day that a fixed daily reference time is
    standard practice for this kind of scan).
    """
    import datetime
    results = []
    start = datetime.date(start_year, start_month, start_day)
    for i in range(num_days):
        d = start + datetime.timedelta(days=i)
        jd_ut = julian_day_utc(d.year, d.month, d.day, 12, 0, 0)
        day_positions = compute_positions(jd_ut)
        score, hits = score_day_against_natal(day_positions, natal_positions, lens, natal_houses)
        results.append({"date": d.isoformat(), "score": score, "hits": hits})
    return sorted(results, key=lambda r: r["score"], reverse=True)


# --- Query routing -------------------------------------------------------
# Lightweight keyword classifier standing in for what a production system
# would do with an LLM call (Claude, given the free-text question, picks
# the lens with far more nuance than keyword matching ever could -- this
# is a placeholder so the pipeline is demonstrable end-to-end here).
LENS_KEYWORDS = {
    "money": ["money", "income", "invest", "raise", "price", "sell", "earn", "financ"],
    "career": ["launch", "career", "business", "job", "promotion", "work", "hire", "quit"],
    "relationships": ["relationship", "partner", "date", "love", "marry", "breakup", "ex"],
}


def route_question_to_lens(question_text):
    """Naive keyword router. Returns the lens name and which keyword matched."""
    q = question_text.lower()
    for lens, keywords in LENS_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return lens, kw
    return "timing", None


# --- Reading assembly ------------------------------------------------------
# Turns a scored day's raw aspect hits into the three-part structure that
# tested well: a direct verdict, the strongest favorable driver ("why"),
# and the strongest tense factor if one exists ("what's off"). This is a
# STARTER phrase library, not a finished one -- only a handful of specific
# planet/aspect combos have real written lines; everything else falls back
# to a generic template. Scaling this to feel consistently specific across
# every combination is real, ongoing writing work, not a one-time build.
WHY_PHRASES = {
    ("Sun", "trine"): "The Sun's lined up well with your natal {n_planet} today — steady, low-effort support around {n_planet_house_note}.",
    ("Sun", "sextile"): "The Sun's giving your natal {n_planet} a light boost — a small, easy opening around {n_planet_house_note}.",
    ("Moon", "trine"): "The Moon's trine your natal {n_planet} today — nothing forced, just easy going around {n_planet_house_note}.",
    ("Moon", "sextile"): "The Moon's lightly favoring your natal {n_planet} — a gentle, low-key opening around {n_planet_house_note}.",
    ("Mercury", "trine"): "Mercury's flowing well with your natal {n_planet} — conversations and decisions around {n_planet_house_note} should come easier than usual.",
    ("Mercury", "sextile"): "Mercury's lightly favoring your natal {n_planet} — a good window to talk through anything involving {n_planet_house_note}.",
    ("Venus", "trine"): "Venus is forming an easy angle to your natal {n_planet} today, right around {n_planet_house_note}.",
    ("Venus", "sextile"): "Venus is lightly favoring your natal {n_planet} — a small opening around {n_planet_house_note}.",
    ("Mars", "trine"): "Mars is giving your natal {n_planet} some real momentum today — good energy for anything involving {n_planet_house_note}.",
    ("Mars", "sextile"): "Mars is lightly energizing your natal {n_planet} — a decent window to actually move on {n_planet_house_note}.",
    ("Jupiter", "trine"): "Jupiter's trine your natal {n_planet} — this is the kind of day that tends to work out better than expected around {n_planet_house_note}.",
    ("Jupiter", "sextile"): "Jupiter's lightly favoring your natal {n_planet} — a decent-sized opening around {n_planet_house_note}.",
    ("Saturn", "trine"): "Saturn's actually supporting your natal {n_planet} right now — less exciting than favorable, but solid, around {n_planet_house_note}.",
    ("Saturn", "sextile"): "Saturn's lightly backing your natal {n_planet} — a steady, unglamorous kind of support around {n_planet_house_note}.",
    ("Uranus", "trine"): "Uranus is trine your natal {n_planet} — a good day for something a little unexpected to work out around {n_planet_house_note}.",
    ("Uranus", "sextile"): "Uranus is lightly sparking your natal {n_planet} — a small opening to try something different around {n_planet_house_note}.",
    ("Neptune", "trine"): "Neptune's trine your natal {n_planet} — good intuition day, especially around {n_planet_house_note}. Trust the hunch.",
    ("Neptune", "sextile"): "Neptune's lightly favoring your natal {n_planet} — a soft, dreamy opening around {n_planet_house_note}.",
    ("Pluto", "trine"): "Pluto's trine your natal {n_planet} — real, lasting change is easier to make around {n_planet_house_note} right now.",
    ("Pluto", "sextile"): "Pluto's lightly supporting your natal {n_planet} — a small chance to shift something around {n_planet_house_note} for good.",
    ("North Node", "trine"): "The North Node's trine your natal {n_planet} — this pulls you toward where you're actually headed, around {n_planet_house_note}.",
    ("North Node", "sextile"): "The North Node's lightly favoring your natal {n_planet} — a nudge in the right direction around {n_planet_house_note}.",
}
WHAT_S_OFF_PHRASES = {
    ("Sun", "square"): "The Sun's squaring your natal {n_planet} — a little friction around {n_planet_house_note}, more annoying than serious.",
    ("Sun", "opposition"): "The Sun's opposing your natal {n_planet} — expect some pull between what you want and what's actually in front of you around {n_planet_house_note}.",
    ("Moon", "square"): "The Moon's squaring your natal {n_planet} — moodier than usual around {n_planet_house_note}, probably won't last past today.",
    ("Moon", "opposition"): "The Moon's opposing your natal {n_planet} — you might feel pulled in two directions around {n_planet_house_note} today.",
    ("Mercury", "square"): "Mercury's squaring your natal {n_planet} — miscommunication risk around {n_planet_house_note}. Reread anything before you send it.",
    ("Mercury", "opposition"): "Mercury's opposing your natal {n_planet} — you and someone else may just be seeing {n_planet_house_note} differently today. Worth double-checking before assuming.",
    ("Venus", "square"): "Venus is squaring your natal {n_planet} — a little tension around {n_planet_house_note}, nothing that won't pass.",
    ("Venus", "opposition"): "Venus is opposing your natal {n_planet} — a pull between what feels good and what's actually good for {n_planet_house_note}.",
    ("Mars", "square"): "Mars squares your natal {n_planet} — short-fuse energy around {n_planet_house_note}. Don't force it if it's not flowing.",
    ("Mars", "opposition"): "Mars opposes your natal {n_planet} — real risk of a power struggle around {n_planet_house_note}. Pick your moment.",
    ("Jupiter", "square"): "Jupiter's squaring your natal {n_planet} — easy to overdo it around {n_planet_house_note} today. Good day to double-check the math before committing.",
    ("Jupiter", "opposition"): "Jupiter's opposing your natal {n_planet} — a temptation to overpromise around {n_planet_house_note}. Worth sitting with it a beat longer.",
    ("Saturn", "square"): "Saturn squares your natal {n_planet} — expect some friction or delay around {n_planet_house_note}, not a hard no.",
    ("Saturn", "opposition"): "Saturn opposes your natal {n_planet} — more of a gut check than a real obstacle around {n_planet_house_note}.",
    ("Uranus", "square"): "Uranus squares your natal {n_planet} — something around {n_planet_house_note} could shift without warning today. Roll with it if it does.",
    ("Uranus", "opposition"): "Uranus opposes your natal {n_planet} — a sudden pull toward doing something different around {n_planet_house_note}. Sleep on the big version of it.",
    ("Neptune", "square"): "Neptune squares your natal {n_planet} — things around {n_planet_house_note} might feel foggier than they actually are. Get the specifics in writing.",
    ("Neptune", "opposition"): "Neptune opposes your natal {n_planet} — easy to see what you want to see around {n_planet_house_note} today instead of what's actually there.",
    ("Pluto", "square"): "Pluto squares your natal {n_planet} — intense, not necessarily bad, around {n_planet_house_note}. Give it a day before reacting.",
    ("Pluto", "opposition"): "Pluto opposes your natal {n_planet} — a power dynamic around {n_planet_house_note} might come to a head. Stay aware of it, don't force a resolution today.",
    ("North Node", "square"): "The North Node squares your natal {n_planet} — a little friction between where you're comfortable and where you're headed, around {n_planet_house_note}.",
    ("North Node", "opposition"): "The North Node opposes your natal {n_planet} — old habits around {n_planet_house_note} might feel extra tempting today. Worth noticing, not necessarily following.",
}
HOUSE_NOTES = {
    1: "your sense of self", 2: "your money and what you value",
    3: "how you think and communicate", 4: "home and where you live",
    5: "creativity and what you enjoy", 6: "your daily work and routine",
    7: "partnership", 8: "shared resources and deep change",
    9: "growth, travel, and belief", 10: "your public path and career",
    11: "community and long-term goals", 12: "rest and what's behind the scenes",
}


def _phrase_for(hit, natal_houses, phrase_bank):
    key = (hit["transiting"], hit["aspect"])
    house_note = ""
    if natal_houses:
        # requires the natal planet's longitude, not present on the hit itself --
        # caller passes it in via hit["natal_longitude"] when available
        pass
    template = phrase_bank.get(key)
    if template:
        return template.format(
            t_planet=hit["transiting"], n_planet=hit["natal"],
            n_planet_house_note=hit.get("house_note", "worth paying attention to"),
        )
    # generic fallback -- honest that it's less specific than a written line
    verb = {"trine": "flowing well with", "sextile": "lightly favoring",
            "square": "creating friction with", "opposition": "pulling against",
            "conjunction": "sitting right on top of"}[hit["aspect"]]
    return f"{hit['transiting']} is {verb} your natal {hit['natal']} right now."


def generate_reading(day_result, natal_positions, natal_houses=None):
    """
    Takes one day's result from scan_date_range (score + hits) and returns
    the three-part structure: verdict, why, whats_off. whats_off is None
    on a genuinely clean day -- we don't invent tension that isn't there.
    """
    score = day_result["score"]
    hits = day_result["hits"]

    if score >= 4:
        verdict = "This one leans favorable."
    elif score >= 1:
        verdict = "Mild lean toward yes, nothing dramatic either way."
    elif score > -1:
        verdict = "Genuinely mixed -- no strong pull in either direction."
    else:
        verdict = "This one leans toward waiting."

    favorable_hits = [h for h in hits if h["aspect"] in FAVORABLE]
    tense_hits = [h for h in hits if h["aspect"] in TENSE]

    # attach a house note to the natal side of each favorable hit, if we
    # have house data, so the "why" line can say WHAT area of life it's about
    for h in favorable_hits + tense_hits:
        if natal_houses and h["natal"] in natal_positions:
            house_num = which_house(natal_positions[h["natal"]]["longitude"], natal_houses)
            h["house_note"] = HOUSE_NOTES.get(house_num, "an area worth paying attention to")

    why = None
    if favorable_hits:
        top_favorable = max(favorable_hits, key=lambda h: h["weight"])
        why = _phrase_for(top_favorable, natal_houses, WHY_PHRASES)

    whats_off = None
    if tense_hits:
        top_tense = max(tense_hits, key=lambda h: h["weight"])
        whats_off = _phrase_for(top_tense, natal_houses, WHAT_S_OFF_PHRASES)

    return {"date": day_result["date"], "score": score,
            "verdict": verdict, "why": why, "whats_off": whats_off}


# --- Question routing (real version) ---------------------------------------
# Replaces the keyword-router placeholder with an actual classification
# call. Requires ANTHROPIC_API_KEY set in the real deployment's environment
# -- this sandbox has no production key, so this function is untested
# against a live model here (see the mocked test harness below instead,
# which proves the confidence-threshold branching logic independent of
# what the model actually returns).

import json
import os
import urllib.request

CONFIDENCE_THRESHOLD = 0.7

CLASSIFY_SYSTEM_PROMPT = """You classify a user's question into one of these
lenses for an astrology app: money, career, relationships, timing (general/other).

Return ONLY valid JSON, no other text:
{"lens": "<top guess>", "confidence": <0.0-1.0>, "second_guess": "<lens or null>"}

confidence should be LOW (below 0.6) if the question is genuinely ambiguous
between two lenses, or doesn't clearly fit any of them (e.g. a housing/home
question isn't cleanly any of the four -- score it low and let the app ask
the user directly rather than guessing)."""


def classify_question_live(question_text, api_key=None):
    """
    Real classification call. Set ANTHROPIC_API_KEY in the environment
    (or pass api_key) before calling this in production.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set -- can't make a live call")

    body = json.dumps({
        "model": "claude-haiku-4-5-20251001",  # cheap/fast, right-sized for classification
        "max_tokens": 100,
        "system": CLASSIFY_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": question_text}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    return json.loads(result["content"][0]["text"])


def route_with_confidence(question_text, classify_fn=classify_question_live):
    """
    The actual app-facing entry point. classify_fn is swappable so this
    same branching logic can be tested with a mock instead of a live call.
    Returns either a direct route or a clarify-screen instruction.
    """
    result = classify_fn(question_text)
    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        return {"action": "route_directly", "lens": result["lens"]}
    else:
        options = [result["lens"]]
        if result.get("second_guess"):
            options.append(result["second_guess"])
        return {"action": "show_clarify", "question": question_text, "options": options}


# --- Full pipeline ----------------------------------------------------------
def handle_question(question_text, natal_chart, lat, lon,
                     start_year, start_month, start_day, num_days=30,
                     classify_fn=classify_question_live, house_system="placidus"):
    """
    The single entry point tying every piece together: routes the
    question, and if confident, scans the date range through that lens
    and returns a fully assembled reading. If not confident, returns
    the clarify-screen instruction instead so the app can show the
    tappable options -- never a guessed answer.
    """
    routing = route_with_confidence(question_text, classify_fn)
    if routing["action"] == "show_clarify":
        return routing

    natal_houses = natal_chart["houses_and_angles"][house_system]["houses"]
    results = scan_date_range(
        natal_chart["positions"], start_year, start_month, start_day, num_days,
        lat, lon, lens=routing["lens"], natal_houses=natal_houses,
    )
    top_day = results[0]
    reading = generate_reading(top_day, natal_chart["positions"], natal_houses)
    return {"action": "show_reading", "lens": routing["lens"],
            "question": question_text, "reading": reading}


# --- Astrocartography --------------------------------------------------
# Where each planet's angular lines (MC/IC meridians, AC/DC curves) cross
# the globe. MC/IC are straightforward meridians; AC/DC require actual
# spherical trig since whether a planet is on the horizon depends on both
# longitude AND latitude. This is the one piece of the platform that has
# nothing to do with the natal-chart math already built -- separate
# calculation entirely, as flagged back when this was first scoped.

EQ_FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED | swe.FLG_EQUATORIAL


def compute_astrocartography_lines(jd_ut, lat_range=(-66, 66), lat_step=2):
    """
    Returns, per planet: MC/IC longitude (simple meridians) and AC/DC as
    a list of (latitude, longitude) points tracing the curved rising/
    setting lines. Planets that are circumpolar at a given latitude (never
    rise or set there) are skipped for that latitude -- expected behavior,
    not a bug.
    """
    gst_hours = swe.sidtime(jd_ut)
    gst_deg = gst_hours * 15

    lines = {}
    for name, code in PLANETS.items():
        try:
            xx, _ = swe.calc_ut(jd_ut, code, EQ_FLAGS)
        except swe.Error:
            continue
        ra, dec = xx[0], xx[1]  # equatorial: right ascension, declination (degrees)

        mc_lon = _normalize_lon(ra - gst_deg)
        ic_lon = _normalize_lon(mc_lon + 180)

        ac_curve, dc_curve = [], []
        lat_vals = range(lat_range[0], lat_range[1] + 1, lat_step)
        for glat in lat_vals:
            tan_product = math.tan(math.radians(glat)) * math.tan(math.radians(dec))
            if abs(tan_product) > 1:
                continue  # circumpolar at this latitude -- no rise/set line here
            H = math.degrees(math.acos(-tan_product))
            ac_lon = _normalize_lon(ra - gst_deg - H)
            dc_lon = _normalize_lon(ra - gst_deg + H)
            ac_curve.append((glat, ac_lon))
            dc_curve.append((glat, dc_lon))

        lines[name] = {"mc_lon": round(mc_lon, 2), "ic_lon": round(ic_lon, 2),
                        "ac_curve": ac_curve, "dc_curve": dc_curve}
    return lines


def _normalize_lon(lon):
    """Wrap a longitude to -180..180."""
    lon = lon % 360
    return lon - 360 if lon > 180 else lon


def _curve_lon_at_lat(curve, query_lat):
    """Interpolate a sampled AC/DC curve to find its longitude at an exact latitude."""
    if not curve:
        return None
    closest = min(curve, key=lambda pt: abs(pt[0] - query_lat))
    return closest[1]


def check_location_influence(lines, query_lat, query_lon, orb_degrees=6):
    """
    For a query location, finds every planetary line within orb, sorted
    by closeness. Distance is in degrees of longitude at that latitude --
    the standard astrocartography convention.
    """
    hits = []
    for planet, data in lines.items():
        for line_type in ("mc_lon", "ic_lon"):
            dist = abs(_lon_diff(query_lon, data[line_type]))
            if dist <= orb_degrees:
                hits.append({"planet": planet, "line": line_type.upper().replace("_LON", ""),
                             "distance_deg": round(dist, 2)})
        for line_type, curve in (("AC", data["ac_curve"]), ("DC", data["dc_curve"])):
            line_lon = _curve_lon_at_lat(curve, query_lat)
            if line_lon is not None:
                dist = abs(_lon_diff(query_lon, line_lon))
                if dist <= orb_degrees:
                    hits.append({"planet": planet, "line": line_type, "distance_deg": round(dist, 2)})
    return sorted(hits, key=lambda h: h["distance_deg"])


def _lon_diff(lon1, lon2):
    d = (lon1 - lon2) % 360
    return d - 360 if d > 180 else d


# --- Synastry ----------------------------------------------------------
# Comparing two charts against each other -- for relationship compatibility
# OR for founder-to-business comparison, since both are just "two already-
# computed charts, find the aspects between them." No new astronomical
# calculation needed here, just cross-referencing chart_a's positions
# against chart_b's using the same find_aspect() logic used for transits.

def compute_synastry(chart_a_positions, chart_b_positions, label_a="A", label_b="B"):
    """
    Every aspect between person/entity A's planets and person/entity B's
    planets. Unlike a transit scan, there's no "weight by speed" here --
    both charts are fixed points in time, so every hit is treated as
    equally real rather than fast-vs-slow.
    """
    hits = []
    for a_name, a_data in chart_a_positions.items():
        if a_name == "_skipped":
            continue
        for b_name, b_data in chart_b_positions.items():
            if b_name == "_skipped":
                continue
            result = find_aspect(a_data["longitude"], b_data["longitude"])
            if result:
                aspect_name, exactness = result
                hits.append({
                    f"{label_a}_planet": a_name, f"{label_b}_planet": b_name,
                    "aspect": aspect_name, "orb": exactness,
                })
    return sorted(hits, key=lambda h: h["orb"])


def compute_chart(year, month, day, hour, minute, lat, lon, unknown_time=False):
    """
    Main entry point. Caller supplies LOCAL birth date/time + coordinates —
    the correct historical UTC offset is resolved automatically.

    If unknown_time=True, defaults to noon local and omits houses/angles
    (they're meaningless without an exact birth time) — planetary signs
    are still valid since most planets don't change sign within a single day.
    """
    if unknown_time:
        hour, minute = 12, 0

    utc_offset_hours, tz_name = resolve_utc_offset(year, month, day, hour, minute, lat, lon)
    jd_ut = julian_day_utc(year, month, day, hour, minute, utc_offset_hours)
    positions = compute_positions(jd_ut)

    result = {
        "julian_day_ut": jd_ut,
        "timezone": tz_name,
        "utc_offset_hours": utc_offset_hours,
        "positions": positions,
        "houses_and_angles": None,
        "time_known": not unknown_time,
    }

    if not unknown_time:
        result["houses_and_angles"] = compute_angles_and_houses(jd_ut, lat, lon)

    return result
