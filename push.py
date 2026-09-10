# push.py—Web push notifications for Estrella.
#
# Deploys to Railway alongside chart_service.py, chart_engine.py, and
# billing.py. Requires these environment variables:
#   VAPID_PRIVATE_KEY—the PEM private key generated for this app
#   VAPID_PUBLIC_KEY —the matching base64url public key (also needed
#                         on the frontend as NEXT_PUBLIC_VAPID_PUBLIC_KEY
#                       —it's the SAME value in both places)
#   VAPID_CONTACT_EMAIL—a real contact email, required by the push
#                         spec so a browser vendor can reach you if a
#                         subscriber's push traffic looks abusive
#   SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY—same as billing.py
#
# What this file does NOT do on its own: decide WHEN to send anything.
# /push/send-daily below is a real, working endpoint that checks
# whether the moon phase actually changed since yesterday and sends a
# notification to everyone subscribed if so—but something has to
# actually CALL that endpoint once a day for this to happen
# automatically. Railway doesn't run scheduled jobs on its own; this
# needs either Railway's own cron feature (if your plan includes it)
# or a free external scheduler (cron-job.org, GitHub Actions on a
# schedule, etc.) configured to hit this endpoint daily. That setup
# step is yours to do—it can't be configured from here.

import os
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from pywebpush import webpush, WebPushException
import json as jsonlib
from supabase import create_client, Client
import chart_engine as ce
import billing

router = APIRouter()

_VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
_VAPID_CLAIMS = {"sub": f"mailto:{os.environ.get('VAPID_CONTACT_EMAIL', '')}"}

_supabase_admin: Client = create_client(
    os.environ.get("SUPABASE_URL", ""),
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
)


def _verify_user(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization[len("Bearer "):]
    try:
        result = _supabase_admin.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    if not result or not result.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return result.user.id


class SubscribeRequest(BaseModel):
    endpoint: str
    p256dh: str
    auth: str


@router.post("/push/subscribe")
def subscribe(req: SubscribeRequest, authorization: str | None = Header(None)):
    user_id = _verify_user(authorization)
    _supabase_admin.table("push_subscriptions").upsert({
        "user_id": user_id, "endpoint": req.endpoint, "p256dh": req.p256dh, "auth_key": req.auth,
    }, on_conflict="endpoint").execute()
    return {"success": True}


class UnsubscribeRequest(BaseModel):
    endpoint: str


@router.post("/push/unsubscribe")
def unsubscribe(req: UnsubscribeRequest, authorization: str | None = Header(None)):
    _verify_user(authorization)
    _supabase_admin.table("push_subscriptions").delete().eq("endpoint", req.endpoint).execute()
    return {"success": True}


def _send_to_device(sub: dict, title: str, body: str, url: str = "/home") -> bool:
    """Sends one notification to one specific subscribed device. Returns
    whether it actually succeeded. A dead/expired subscription (the
    browser un-registered it, the device was reset, etc.) fails with a
    404/410 from the push service—cleaned up here rather than
    retried forever."""
    try:
        webpush(
            subscription_info={
                "endpoint": sub["endpoint"],
                "keys": {"p256dh": sub["p256dh"], "auth": sub["auth_key"]},
            },
            data=jsonlib.dumps({"title": title, "body": body, "url": url}),
            vapid_private_key=_VAPID_PRIVATE_KEY,
            vapid_claims=dict(_VAPID_CLAIMS),
        )
        return True
    except WebPushException as e:
        status = getattr(e.response, "status_code", None)
        if status in (404, 410):
            _supabase_admin.table("push_subscriptions").delete().eq("id", sub["id"]).execute()
        else:
            print(f"[push] failed to send to subscription {sub['id']}: {e}")
        return False


# Same tight-orb threshold across both personal checks below—loose
# enough to actually catch something real, tight enough that this
# doesn't fire most days for most people. A transit that's been in a
# wide, non-notification-worthy orb for weeks shouldn't ping someone's
# phone; something genuinely peaking today should.
_NOTIFY_ORB_DEGREES = 1.0


def _check_progression_sign_change(birth_jd_ut: float, jd_today: float, jd_yesterday: float) -> dict | None:
    """Checks whether the progressed Sun or Moon crossed into a new
    sign between yesterday and today—a genuinely rare event, unlike a
    daily transit check. The progressed Moon changes sign roughly
    every two and a half years; the progressed Sun moves so slowly
    (about a degree a year) that a sign change usually happens only
    once or twice in a lifetime. Checks the Sun first, since a
    progressed Sun sign change is the rarer, more significant of the
    two if both somehow landed on the same day.
    """
    prog_today = ce.compute_progressed_positions(birth_jd_ut, jd_today)
    prog_yesterday = ce.compute_progressed_positions(birth_jd_ut, jd_yesterday)
    for planet in ("Sun", "Moon"):
        sign_today = prog_today["positions"][planet]["sign"]
        sign_yesterday = prog_yesterday["positions"][planet]["sign"]
        if sign_today != sign_yesterday:
            return {"planet": planet, "sign": sign_today}
    return None


def _find_personal_hits(natal_positions: dict, transiting_positions: dict) -> dict | None:
    """Checks one person's own chart against today's transiting outer
    planets (Jupiter through Pluto—the only ones slow and significant
    enough to be worth a same-day alert; fast planets aspect constantly
    and would make this spam). Returns the single tightest real hit, if
    any, distinguishing a genuine "return" (a planet transiting back
    over the exact same natal placement) from any other tight aspect,
    since a return is significant in its own right, not just another
    conjunction.
    """
    best = None
    for t_name, t_data in transiting_positions.items():
        if t_name == "_skipped" or t_name not in ce.OUTER_PLANETS:
            continue
        for n_name, n_data in natal_positions.items():
            if n_name == "_skipped":
                continue
            result = ce.find_aspect(t_data["longitude"], n_data["longitude"])
            if not result:
                continue
            aspect_name, exactness = result
            if exactness > _NOTIFY_ORB_DEGREES:
                continue
            is_return = (t_name == n_name and aspect_name == "conjunction")
            if best is None or exactness < best["exactness"]:
                best = {"transiting": t_name, "natal": n_name, "aspect": aspect_name, "exactness": exactness, "is_return": is_return}
    return best


@router.post("/push/send-event-reminders")
def send_event_reminders(x_cron_secret: str | None = Header(None, alias="X-Cron-Secret")):
    """Checks for calendar event reminders that are due right now and
    sends them. Unlike send_daily above, which only needs to run once
    a day, this needs to be called frequently -- every 5 to 15
    minutes -- since a reminder set for "30 minutes before" is only
    useful if this actually notices within a similar window, not once
    a day. This is a genuinely separate cron job from the one already
    calling send_daily, not a replacement for it -- both need to keep
    running, on their own separate schedules.

    Protected by the same shared-secret pattern as send_daily, for the
    same reason: configure your scheduler to send CRON_SECRET as this
    header too.

    Real, working per-occurrence tracking, not a guess: reminder_sent
    is checked and then set on the specific row that fired, so a
    recurring event's ten different future occurrences each get their
    own independent reminder, and a reminder already sent is never
    sent twice even if this runs again a few minutes later and the
    row hasn't cleared the query window yet.
    """
    expected_secret = os.environ.get("CRON_SECRET")
    if expected_secret and x_cron_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing cron secret")

    from datetime import datetime, timedelta, timezone as dt_timezone
    from zoneinfo import ZoneInfo

    now_utc = datetime.now(dt_timezone.utc)

    # Only ever pending reminders in the first place -- the partial
    # index on (remind_me, reminder_sent) in the schema keeps this
    # fast regardless of how large calendar_events grows overall,
    # since it was built exactly for this query.
    pending = _supabase_admin.table("calendar_events").select(
        "id, user_id, event_date, event_time, title, reminder_offset_minutes"
    ).eq("remind_me", True).eq("reminder_sent", False).execute()
    rows = pending.data or []
    if not rows:
        # Printed on every run, not just when something's actually due
        # -- a real, reported problem this is fixing directly: this
        # endpoint returning 200 OK told Milli nothing about whether it
        # actually found anything, only that the request didn't crash.
        # Visible in Railway's own logs now, no separate check needed.
        print("[push] send-event-reminders: no reminders currently pending at all (remind_me=true, reminder_sent=false)")
        return {"checked": 0, "sent": 0}

    # Each user's own timezone, fetched once in a batch rather than
    # once per row -- same reasoning as send_daily's own preferences
    # batch above. Defaults to UTC for anyone whose row doesn't have
    # one set yet, matching the frontend's own fallback.
    user_ids = list({r["user_id"] for r in rows})
    tz_rows = _supabase_admin.table("users").select("id, preferred_timezone").in_("id", user_ids).execute()
    tz_by_user = {row["id"]: (row.get("preferred_timezone") or "UTC") for row in (tz_rows.data or [])}

    due_ids: list[str] = []
    due_rows: list[dict] = []
    for row in rows:
        if not row.get("event_time"):
            # A time is required in the UI before a reminder can be
            # saved in the first place -- this is a defensive check
            # for a row that somehow has one unset anyway (a manual
            # DB edit, a future bug elsewhere), not a path expected to
            # ever actually run.
            continue
        try:
            tz = ZoneInfo(tz_by_user.get(row["user_id"], "UTC"))
            event_date = datetime.strptime(row["event_date"], "%Y-%m-%d").date()
            event_time = datetime.strptime(row["event_time"][:5], "%H:%M").time()
            event_local = datetime.combine(event_date, event_time, tzinfo=tz)
            event_utc = event_local.astimezone(dt_timezone.utc)
            reminder_moment = event_utc - timedelta(minutes=row["reminder_offset_minutes"] or 0)
            # Printed for every pending row, due or not -- this is the
            # actual, direct answer to "why hasn't my reminder fired,"
            # visible in Railway's logs without needing cron-job.org at
            # all: the exact UTC instants this run computed for the
            # event and its reminder, against the exact UTC instant
            # this run considers "now." If the event's own time zone
            # was somehow wrong (a stale preferred_timezone, or none
            # ever set for a guest-created event), it shows up here as
            # an event_utc that's obviously off from what was actually
            # intended, rather than silently never firing with no
            # visible reason why.
            print(f"[push] event {row['id']} ({row['title']}): event_utc={event_utc.isoformat()}, reminder_moment={reminder_moment.isoformat()}, now={now_utc.isoformat()}")
        except Exception as e:
            print(f"[push] couldn't compute reminder time for event {row['id']}: {e}")
            continue

        # Fires once the reminder moment has arrived, but only within
        # a bounded window after the event itself -- a real, deliberate
        # guard, not an arbitrary number: without this, a cron outage
        # of a few hours (or longer) would come back and fire every
        # single reminder that piled up while it was down, including
        # ones for events that already happened. Two hours past the
        # event's own time is long enough to absorb a normal short
        # outage without ever surfacing a reminder for something
        # that's already over and gone.
        if reminder_moment <= now_utc <= event_utc + timedelta(hours=2):
            due_ids.append(row["id"])
            due_rows.append(row)

    if not due_rows:
        print(f"[push] send-event-reminders: {len(rows)} pending reminder(s) found, none due yet -- see the per-event timing lines above for exactly why")
        return {"checked": len(rows), "sent": 0}

    subs = _supabase_admin.table("push_subscriptions").select(
        "id, user_id, endpoint, p256dh, auth_key"
    ).in_("user_id", list({r["user_id"] for r in due_rows})).execute()
    subs_by_user: dict[str, list[dict]] = {}
    for sub in subs.data or []:
        subs_by_user.setdefault(sub["user_id"], []).append(sub)

    def _describe_offset(minutes: int) -> str:
        # Natural phrasing for the common presets, not a raw number of
        # minutes shown back to the person -- "tomorrow" and "in an
        # hour" read like something a person would actually say;
        # "in 1440 minutes" doesn't. Falls back to plain minutes for
        # anything that doesn't land on a clean unit, which only
        # happens for a genuinely unusual custom value.
        if minutes == 1440:
            return "tomorrow"
        if minutes % 1440 == 0:
            days = minutes // 1440
            return f"in {days} day{'s' if days != 1 else ''}"
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"in {hours} hour{'s' if hours != 1 else ''}"
        return f"in {minutes} minutes"

    sent = 0
    for row in due_rows:
        offset_desc = _describe_offset(row["reminder_offset_minutes"] or 0)
        message = f"{row['title']} tomorrow" if offset_desc == "tomorrow" else f"{row['title']} {offset_desc}"
        user_subs = subs_by_user.get(row["user_id"], [])
        if not user_subs:
            # A due reminder with genuinely nowhere to send it -- most
            # likely notifications were never turned on for this
            # account, or the one subscription that existed expired
            # and was already cleaned up by _send_to_device elsewhere.
            # Printed specifically, not folded into the generic "sent"
            # count, since "0 sent because nothing was due" and "0 sent
            # because it was due but no device to reach" are different
            # problems needing different fixes.
            print(f"[push] event {row['id']} ({row['title']}) is due but user {row['user_id']} has no active push subscription")
            continue
        for sub in user_subs:
            if _send_to_device(sub, "Estrella", message, "/calendar"):
                sent += 1

    # Marked sent regardless of whether a push subscription actually
    # existed to receive it -- someone with reminders configured but
    # no active push subscription (notifications turned off at the OS
    # level, no subscription ever created) shouldn't have this row
    # checked again on every single run forever; the reminder's
    # moment already passed once, and re-checking it indefinitely
    # would just be wasted work with no different outcome next time.
    for event_id in due_ids:
        _supabase_admin.table("calendar_events").update({"reminder_sent": True}).eq("id", event_id).execute()

    return {"checked": len(rows), "due": len(due_rows), "sent": sent}

def send_daily(x_cron_secret: str | None = Header(None, alias="X-Cron-Secret")):
    """The real, comprehensive daily check—not just moon phases.
    Covers, every day:
      - Moon phase changes (global, same for everyone)
      - A retrograde starting today (global)
      - An eclipse today (global)
      - Each subscribed user's OWN chart checked for a genuine return
        or a tight, currently-peaking outer-planet transit (personal,
        different for every person, and the part that was missing
        entirely before this).
      - Each subscribed user's progressed Sun or Moon crossing into a
        new sign since yesterday—a genuinely rare event (the Moon
        roughly every two and a half years, the Sun maybe once or
        twice in a lifetime), not a daily check like the transit one
        above.
    Needs something external calling this once a day—see the module
    docstring above.

    Protected by a shared secret (set CRON_SECRET on Railway and
    configure your scheduler to send it as this header) so this can't
    be triggered by anyone who happens to find the URL—and guarded
    against firing twice on the same day even if it IS called more
    than once, via a unique-per-date row in push_send_log, so an
    accidental duplicate call never means a duplicate notification to
    everyone subscribed.
    """
    expected_secret = os.environ.get("CRON_SECRET")
    if expected_secret and x_cron_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing cron secret")

    from datetime import date, timedelta
    today = date.today()
    yesterday = today - timedelta(days=1)
    jd_today = ce.julian_day_utc(today.year, today.month, today.day, 12, 0, 0)
    jd_yesterday = ce.julian_day_utc(yesterday.year, yesterday.month, yesterday.day, 12, 0, 0)

    try:
        # The unique constraint on send_date does the real work here --
        # if this row already exists for today (a second call landed
        # after the first one succeeded), this insert fails and the
        # exception below stops the whole run before anything sends.
        _supabase_admin.table("push_send_log").insert({
            "send_date": today.isoformat(), "phase": "pending", "sent_count": 0,
        }).execute()
    except Exception:
        return {"sent": 0, "already_ran_today": True}

    # --- Global events, computed once, shared by everyone—kept as
    # separate, labeled lines rather than one joined string, so each
    # category can be independently included or skipped per recipient
    # based on their own notification_prefs below.
    global_events: dict[str, str] = {}

    phase_today = ce.moon_phase(jd_today)["phase"]
    phase_yesterday = ce.moon_phase(jd_yesterday)["phase"]
    if phase_today != phase_yesterday:
        global_events["moon_phase"] = f"{phase_today} today—worth knowing before you make any big moves."

    positions_today = ce.compute_positions(jd_today)
    positions_yesterday = ce.compute_positions(jd_yesterday)
    for planet in ce.RETROGRADE_PLANETS:
        today_retro = positions_today.get(planet, {}).get("retrograde")
        yesterday_retro = positions_yesterday.get(planet, {}).get("retrograde")
        if today_retro and not yesterday_retro:
            # Only one retrograde line per day in practice (multiple
            # planets starting retrograde on the same day is rare), but
            # if it ever happens, the later one simply overwrites --
            # acceptable, since missing a same-day second retrograde
            # notice is a minor loss, not a real bug.
            global_events["retrograde"] = f"{planet} just went retrograde. Buckle up."

    eclipses_today = ce.find_eclipses_in_range(jd_today - 0.5, jd_today + 0.5)
    if eclipses_today:
        global_events["eclipse"] = f"Today's a {eclipses_today[0]['type']} eclipse, and eclipses don't do subtle."

    # --- Housekeeping: revert any expired comped period back to
    # genuinely 'free' status. A comped period (no real Stripe
    # subscription behind it—a referral reward or birthday gift
    # given to someone on the free tier) has nothing that automatically
    # flips its status back once the period ends; PaywallGate already
    # correctly blocks access by comparing today's date to the stored
    # end date, so nobody is ever over-accessing, but the status field
    # itself would otherwise sit there saying "active" forever, which
    # would mislead the admin panel and dead-end a user who looks at
    # their own billing page. Scoped narrowly: only rows with NO real
    # stripe_subscription_id, so this never touches an actual paying
    # subscriber—those transitions are Stripe's own job, already
    # handled correctly by the webhook.
    expired = _supabase_admin.table("users").select("id").is_("stripe_subscription_id", "null") \
        .in_("subscription_status", ["active", "trial"]).lt("subscription_current_period_end", today.isoformat()).execute()
    for row in expired.data or []:
        _supabase_admin.table("users").update({
            "subscription_status": "free", "subscription_current_period_end": None,
        }).eq("id", row["id"]).execute()

    # --- Birthday gifts: a real free month, not just a notification --
    # applies to everyone whose birthday is today regardless of whether
    # they have push enabled, since the gift itself shouldn't depend on
    # having notifications turned on. The notification about it only
    # reaches people who do have push enabled, naturally, since only
    # they're in the per-subscriber loop below.
    birthday_messages: dict[str, str] = {}
    birthday_charts = _supabase_admin.table("charts").select("user_id").eq("chart_type", "personal") \
        .eq("birth_month", today.month).eq("birth_day", today.day).execute()
    for chart_row in birthday_charts.data or []:
        user_id = chart_row["user_id"]
        user_row = _supabase_admin.table("users").select(
            "subscription_status, stripe_subscription_id, subscription_current_period_end, last_birthday_gift_year"
        ).eq("id", user_id).single().execute()
        if not user_row.data or user_row.data.get("last_birthday_gift_year") == today.year:
            continue  # already gifted this year, or no matching user row—skip either way
        try:
            billing._grant_free_month(user_id, user_row.data)
            _supabase_admin.table("users").update({"last_birthday_gift_year": today.year}).eq("id", user_id).execute()
            # _grant_free_month silently no-ops for a lifetime member --
            # correct, since they don't need or benefit from a free
            # month—but the message here was being sent unconditionally,
            # meaning a lifetime member would get a birthday notification
            # falsely claiming a free month had just been added when
            # nothing was actually granted. Checked explicitly here
            # instead of relying on the shared function to report back
            # whether it did anything, since changing its return value
            # would also affect the referral reward path that calls it.
            if user_row.data.get("subscription_status") == "lifetime":
                birthday_messages[user_id] = "Happy birthday. Go do something that actually feels like you."
            else:
                birthday_messages[user_id] = "Happy birthday! We snuck a free month onto your account. Enjoy."
        except Exception as e:
            print(f"[push] couldn't grant birthday gift for user {user_id}: {e}")

    # --- Personal events, one real chart lookup per unique subscriber ---
    subs = _supabase_admin.table("push_subscriptions").select("id, user_id, endpoint, p256dh, auth_key").execute()
    subs_by_user: dict[str, list[dict]] = {}
    for sub in subs.data or []:
        subs_by_user.setdefault(sub["user_id"], []).append(sub)

    # Preferences for every subscriber in one query rather than one
    # lookup per person—defaults to all-true (matching the current,
    # pre-preferences behavior) for anyone whose row somehow doesn't
    # have this set yet, rather than silently sending them nothing.
    DEFAULT_PREFS = {"moon_phase": True, "retrograde": True, "eclipse": True, "personal_transits": True, "birthday": True, "progressions": True}
    prefs_by_user: dict[str, dict] = {}
    if subs_by_user:
        prefs_rows = _supabase_admin.table("users").select("id, notification_prefs").in_("id", list(subs_by_user.keys())).execute()
        for row in prefs_rows.data or []:
            prefs_by_user[row["id"]] = {**DEFAULT_PREFS, **(row.get("notification_prefs") or {})}

    sent = 0
    for user_id, user_subs in subs_by_user.items():
        prefs = prefs_by_user.get(user_id, DEFAULT_PREFS)

        included_global = [line for key, line in global_events.items() if prefs.get(key, True)]

        personal_line = ""
        progression_line = ""
        needs_chart = prefs.get("personal_transits", True) or prefs.get("progressions", True)
        if needs_chart:
            try:
                chart_row = _supabase_admin.table("charts").select("computed_data").eq("user_id", user_id).eq("chart_type", "personal").single().execute()
                if chart_row.data:
                    if prefs.get("personal_transits", True):
                        hit = _find_personal_hits(chart_row.data["computed_data"]["positions"], positions_today)
                        if hit:
                            if hit["is_return"]:
                                personal_line = f"Your {hit['transiting']} Return is happening right now—bigger deal than it sounds."
                            else:
                                personal_line = f"Transiting {hit['transiting']} is in a tight {hit['aspect']} with your natal {hit['natal']} today. Pay attention."
                    if prefs.get("progressions", True):
                        birth_jd_ut = chart_row.data["computed_data"].get("julian_day_ut")
                        if birth_jd_ut:
                            prog_hit = _check_progression_sign_change(birth_jd_ut, jd_today, jd_yesterday)
                            if prog_hit:
                                progression_line = f"Your progressed {prog_hit['planet']} just moved into {prog_hit['sign']}—a real shift most people never even notice happening."
            except Exception as e:
                print(f"[push] couldn't check personal transits or progressions for user {user_id}: {e}")

        # The gift itself was already granted unconditionally above,
        # regardless of preferences—this only controls whether this
        # person also gets pinged about it.
        birthday_line = birthday_messages.get(user_id, "") if prefs.get("birthday", True) else ""

        message = " ".join(filter(None, [*included_global, personal_line, progression_line, birthday_line])).strip()
        if not message:
            continue  # genuinely nothing worth a notification today for this person—stay silent, don't send noise

        for sub in user_subs:
            if _send_to_device(sub, "Estrella", message, "/billing" if birthday_line else ("/moon-phases" if not personal_line else "/ask")):
                sent += 1

    _supabase_admin.table("push_send_log").update({"phase": phase_today, "sent_count": sent}).eq("send_date", today.isoformat()).execute()
    return {"sent": sent, "phase": phase_today, "global_events": list(global_events.values()), "birthdays_gifted": len(birthday_messages)}
