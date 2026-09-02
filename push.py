# push.py -- Web push notifications for Estrella.
#
# Deploys to Railway alongside chart_service.py, chart_engine.py, and
# billing.py. Requires these environment variables:
#   VAPID_PRIVATE_KEY  -- the PEM private key generated for this app
#   VAPID_PUBLIC_KEY   -- the matching base64url public key (also needed
#                         on the frontend as NEXT_PUBLIC_VAPID_PUBLIC_KEY
#                         -- it's the SAME value in both places)
#   VAPID_CONTACT_EMAIL -- a real contact email, required by the push
#                         spec so a browser vendor can reach you if a
#                         subscriber's push traffic looks abusive
#   SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY -- same as billing.py
#
# What this file does NOT do on its own: decide WHEN to send anything.
# /push/send-daily below is a real, working endpoint that checks
# whether the moon phase actually changed since yesterday and sends a
# notification to everyone subscribed if so -- but something has to
# actually CALL that endpoint once a day for this to happen
# automatically. Railway doesn't run scheduled jobs on its own; this
# needs either Railway's own cron feature (if your plan includes it)
# or a free external scheduler (cron-job.org, GitHub Actions on a
# schedule, etc.) configured to hit this endpoint daily. That setup
# step is yours to do -- it can't be configured from here.

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
    404/410 from the push service -- cleaned up here rather than
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


# Same tight-orb threshold across both personal checks below -- loose
# enough to actually catch something real, tight enough that this
# doesn't fire most days for most people. A transit that's been in a
# wide, non-notification-worthy orb for weeks shouldn't ping someone's
# phone; something genuinely peaking today should.
_NOTIFY_ORB_DEGREES = 1.0


def _find_personal_hits(natal_positions: dict, transiting_positions: dict) -> dict | None:
    """Checks one person's own chart against today's transiting outer
    planets (Jupiter through Pluto -- the only ones slow and significant
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


@router.post("/push/send-daily")
def send_daily(x_cron_secret: str | None = Header(None, alias="X-Cron-Secret")):
    """The real, comprehensive daily check -- not just moon phases.
    Covers, every day:
      - Moon phase changes (global, same for everyone)
      - A retrograde starting today (global)
      - An eclipse today (global)
      - Each subscribed user's OWN chart checked for a genuine return
        or a tight, currently-peaking outer-planet transit (personal,
        different for every person, and the part that was missing
        entirely before this).
    Needs something external calling this once a day -- see the module
    docstring above.

    Protected by a shared secret (set CRON_SECRET on Railway and
    configure your scheduler to send it as this header) so this can't
    be triggered by anyone who happens to find the URL -- and guarded
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

    # --- Global events, computed once, shared by everyone -- kept as
    # separate, labeled lines rather than one joined string, so each
    # category can be independently included or skipped per recipient
    # based on their own notification_prefs below.
    global_events: dict[str, str] = {}

    phase_today = ce.moon_phase(jd_today)["phase"]
    phase_yesterday = ce.moon_phase(jd_yesterday)["phase"]
    if phase_today != phase_yesterday:
        global_events["moon_phase"] = f"It's a {phase_today} today."

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
            global_events["retrograde"] = f"{planet} turns retrograde today."

    eclipses_today = ce.find_eclipses_in_range(jd_today - 0.5, jd_today + 0.5)
    if eclipses_today:
        global_events["eclipse"] = f"Today's a {eclipses_today[0]['type']} eclipse."

    # --- Housekeeping: revert any expired comped period back to
    # genuinely 'free' status. A comped period (no real Stripe
    # subscription behind it -- a referral reward or birthday gift
    # given to someone on the free tier) has nothing that automatically
    # flips its status back once the period ends; PaywallGate already
    # correctly blocks access by comparing today's date to the stored
    # end date, so nobody is ever over-accessing, but the status field
    # itself would otherwise sit there saying "active" forever, which
    # would mislead the admin panel and dead-end a user who looks at
    # their own billing page. Scoped narrowly: only rows with NO real
    # stripe_subscription_id, so this never touches an actual paying
    # subscriber -- those transitions are Stripe's own job, already
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
            continue  # already gifted this year, or no matching user row -- skip either way
        try:
            billing._grant_free_month(user_id, user_row.data)
            _supabase_admin.table("users").update({"last_birthday_gift_year": today.year}).eq("id", user_id).execute()
            # _grant_free_month silently no-ops for a lifetime member --
            # correct, since they don't need or benefit from a free
            # month -- but the message here was being sent unconditionally,
            # meaning a lifetime member would get a birthday notification
            # falsely claiming a free month had just been added when
            # nothing was actually granted. Checked explicitly here
            # instead of relying on the shared function to report back
            # whether it did anything, since changing its return value
            # would also affect the referral reward path that calls it.
            if user_row.data.get("subscription_status") == "lifetime":
                birthday_messages[user_id] = "Happy birthday!"
            else:
                birthday_messages[user_id] = "Happy birthday! We added a free month to your account as a gift."
        except Exception as e:
            print(f"[push] couldn't grant birthday gift for user {user_id}: {e}")

    # --- Personal events, one real chart lookup per unique subscriber ---
    subs = _supabase_admin.table("push_subscriptions").select("id, user_id, endpoint, p256dh, auth_key").execute()
    subs_by_user: dict[str, list[dict]] = {}
    for sub in subs.data or []:
        subs_by_user.setdefault(sub["user_id"], []).append(sub)

    # Preferences for every subscriber in one query rather than one
    # lookup per person -- defaults to all-true (matching the current,
    # pre-preferences behavior) for anyone whose row somehow doesn't
    # have this set yet, rather than silently sending them nothing.
    DEFAULT_PREFS = {"moon_phase": True, "retrograde": True, "eclipse": True, "personal_transits": True, "birthday": True}
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
        if prefs.get("personal_transits", True):
            try:
                chart_row = _supabase_admin.table("charts").select("computed_data").eq("user_id", user_id).eq("chart_type", "personal").single().execute()
                if chart_row.data:
                    hit = _find_personal_hits(chart_row.data["computed_data"]["positions"], positions_today)
                    if hit:
                        if hit["is_return"]:
                            personal_line = f"Your {hit['transiting']} Return is happening right now."
                        else:
                            personal_line = f"Transiting {hit['transiting']} is forming a tight {hit['aspect']} to your natal {hit['natal']} today."
            except Exception as e:
                print(f"[push] couldn't check personal transits for user {user_id}: {e}")

        # The gift itself was already granted unconditionally above,
        # regardless of preferences -- this only controls whether this
        # person also gets pinged about it.
        birthday_line = birthday_messages.get(user_id, "") if prefs.get("birthday", True) else ""

        message = " ".join(filter(None, [*included_global, personal_line, birthday_line])).strip()
        if not message:
            continue  # genuinely nothing worth a notification today for this person -- stay silent, don't send noise

        for sub in user_subs:
            if _send_to_device(sub, "Estrella", message, "/billing" if birthday_line else ("/moon-phases" if not personal_line else "/ask")):
                sent += 1

    _supabase_admin.table("push_send_log").update({"phase": phase_today, "sent_count": sent}).eq("send_date", today.isoformat()).execute()
    return {"sent": sent, "phase": phase_today, "global_events": list(global_events.values()), "birthdays_gifted": len(birthday_messages)}
