# billing.py—Stripe subscription handling for Estrella.
#
# Deploys to Railway alongside chart_service.py and chart_engine.py.
# Requires these environment variables to be set on Railway:
#   STRIPE_SECRET_KEY     —from the Stripe dashboard (Developers > API keys)
#   STRIPE_WEBHOOK_SECRET —from the webhook endpoint you create in Stripe
#                              (Developers > Webhooks > add endpoint), NOT the
#                              same as the secret key
#   STRIPE_PRICE_MONTHLY  —the Price ID (starts with "price_") for the
#                              $19.99/month plan, created in Stripe's dashboard
#   STRIPE_PRICE_YEARLY   —the Price ID for the $222/year plan
#   SUPABASE_URL          —same value already used on the frontend
#   SUPABASE_SERVICE_ROLE_KEY—the service role key (not the anon key) --
#                              needed here because webhook events come
#                              directly from Stripe with no user login at
#                              all, so this can't rely on a user's own
#                              session the way the frontend does
#   APP_BASE_URL          —the deployed frontend URL (e.g.
#                              https://astro-app-eight-orpin.vercel.app),
#                              used to build the redirect URLs Stripe sends
#                              people back to after checkout or the portal
#
# Nothing in this file trusts a client-supplied user ID for anything
# billing-related—every checkout/portal request is verified against a
# real Supabase session token first. Admin endpoints go one step further:
# verified session AND is_admin = true on that specific account, a flag
# never exposed anywhere in this app's own signup or profile UI.

import os
import datetime
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel
import stripe
from supabase import create_client, Client

router = APIRouter()

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
_PRICE_IDS = {
    "monthly": os.environ.get("STRIPE_PRICE_MONTHLY"),
    "yearly": os.environ.get("STRIPE_PRICE_YEARLY"),
}
_APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://astro-app-eight-orpin.vercel.app")

_supabase_admin: Client = create_client(
    os.environ.get("SUPABASE_URL", ""),
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
)


def _verify_user(authorization: str | None) -> str:
    """Takes the raw Authorization header, returns the real Supabase user
    id behind it, or raises 401. Asks Supabase's own auth service whether
    the token is genuinely valid, so there's no way to fake being a
    different user by sending a different id in the request body."""
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


def _verify_admin(authorization: str | None) -> str:
    """Same real session verification as _verify_user, plus one more
    check: the caller's own row must have is_admin = true. That flag is
    never set anywhere in the app's own UI—the only way it becomes
    true is you setting it directly in Supabase's table editor."""
    user_id = _verify_user(authorization)
    row = _supabase_admin.table("users").select("is_admin").eq("id", user_id).single().execute()
    if not row.data or not row.data.get("is_admin"):
        # Deliberately the same 404 an unknown route would give, not a
        # 403—a 403 confirms "something real exists here, you're
        # just not allowed," which is itself information worth not
        # handing out to someone who isn't you.
        raise HTTPException(status_code=404, detail="Not found")
    return user_id


def _log_admin_action(email: str, action: str, details: dict | None = None) -> None:
    """Best-effort audit trail—a failure to log should never block
    the actual action from completing, so this deliberately swallows
    its own errors rather than raising."""
    try:
        _supabase_admin.table("admin_grant_log").insert({
            "target_email": email, "action": action, "details": details or {},
        }).execute()
    except Exception as e:
        print(f"[admin log] failed to record action (action still applied): {e}")


def _get_or_create_stripe_customer(user_id: str, email: str | None) -> str:
    row = _supabase_admin.table("users").select("stripe_customer_id").eq("id", user_id).single().execute()
    customer_id = row.data.get("stripe_customer_id") if row.data else None
    if customer_id:
        return customer_id
    customer = stripe.Customer.create(email=email, metadata={"supabase_user_id": user_id})
    _supabase_admin.table("users").update({"stripe_customer_id": customer.id}).eq("id", user_id).execute()
    return customer.id


class CreateCheckoutRequest(BaseModel):
    plan: str  # "monthly" | "yearly"


def _get_or_create_referral_code(user_id: str) -> str:
    row = _supabase_admin.table("users").select("referral_code").eq("id", user_id).single().execute()
    existing = row.data.get("referral_code") if row.data else None
    if existing:
        return existing
    import random, string
    # Generated lazily and retried on the rare collision rather than
    # reserved for every user up front—most people will never look
    # at their own referral code at all.
    for _ in range(5):
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
        try:
            _supabase_admin.table("users").update({"referral_code": code}).eq("id", user_id).execute()
            return code
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Couldn't generate a referral code—try again")


def _grant_free_month(user_id: str, ref_row: dict) -> None:
    """The actual comp-granting logic, extracted so both a referral
    reward and a birthday gift use the exact same, single, already-
    tested path rather than two parallel copies that could quietly
    drift apart. ref_row needs subscription_status, stripe_subscription_id,
    and subscription_current_period_end already selected by the caller.
    """
    if ref_row.get("subscription_status") == "lifetime":
        # A lifetime member's status is never touched by any gift or
        # reward, full stop—independent of whatever
        # stripe_subscription_id happens to contain, so it holds even
        # if that field is ever stale for any reason.
        return

    if ref_row.get("stripe_subscription_id"):
        # A real, currently billing subscription—push the next charge
        # out by 30 days from wherever it currently sits, the same
        # native Stripe mechanism (trial_end on an already-active
        # subscription) used elsewhere for a genuinely free period with
        # no charge attempted until the new date arrives. Stripe itself
        # automatically attempts the charge and flips the subscription
        # back to active (or past_due if it fails) once that date
        # arrives—the existing customer.subscription.updated webhook
        # handler already picks that transition up correctly, so
        # nothing extra is needed for a real subscriber to "resume."
        base = ref_row.get("subscription_current_period_end")
        base_dt = datetime.datetime.fromisoformat(base.replace("Z", "+00:00")) if base else datetime.datetime.now(datetime.timezone.utc)
        new_end = base_dt + datetime.timedelta(days=30)
        stripe.Subscription.modify(ref_row["stripe_subscription_id"], trial_end=int(new_end.timestamp()), proration_behavior="none")
    else:
        # Free tier with no Stripe object at all—a simple comped
        # period, the same direct-Supabase-status approach lifetime
        # and admin resets already use, since there's no real
        # subscription to modify and nothing is actually being charged.
        # Reverting this back to genuinely 'free' once it expires is
        # handled separately, by the daily sweep in push.py—this
        # function only ever grants, never expires anything itself.
        new_end = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        _supabase_admin.table("users").update({
            "subscription_status": "active", "subscription_current_period_end": new_end.isoformat(),
        }).eq("id", user_id).execute()


def _get_card_fingerprint(sub_id: str) -> str | None:
    """Stripe's own fingerprint for the physical card behind a
    subscription—the same value across totally different Stripe
    customers if it's genuinely the same card. Best-effort: returns
    None on any failure rather than raising, since a fraud check that
    can't determine an answer should never be allowed to block a
    legitimate reward or crash the webhook that's also updating the
    referred person's own subscription status.
    """
    try:
        sub = stripe.Subscription.retrieve(sub_id, expand=["default_payment_method"])
        pm = sub.get("default_payment_method")
        if pm and pm.get("card"):
            return pm["card"].get("fingerprint")
        # Not every trial subscription has its own default_payment_method
        # set—falls back to the customer's own default payment method,
        # which trial checkouts do require even when nothing is charged
        # yet.
        customer = stripe.Customer.retrieve(sub["customer"], expand=["invoice_settings.default_payment_method"])
        pm2 = (customer.get("invoice_settings") or {}).get("default_payment_method")
        if pm2 and pm2.get("card"):
            return pm2["card"].get("fingerprint")
    except Exception as e:
        print(f"[referral] couldn't retrieve card fingerprint for {sub_id}, proceeding without a fraud check: {e}")
    return None


_REWARD_CAP_PER_30_DAYS = 5


def _reward_referrer(referred_by_code: str, referred_user_id: str | None = None, referred_email: str | None = None, card_fingerprint: str | None = None) -> None:
    """Called once, the moment a referred person's first subscription
    actually starts—not at signup, since rewarding on signup alone
    (with no payment ever happening) would be trivially abusable.

    Two independent, location-blind fraud checks, neither depending on
    IP address or network at all—the actual exploit here is "one
    person, several fake accounts, referred by themselves," which
    works identically regardless of where the signups happen from:

    1. Same card as the referrer—Stripe's fingerprint catches this
       directly, even across completely different emails and signup
       times.
    2. A rolling cap on how many rewards one account can earn in 30
       days—a hard ceiling on how much any single exploited pattern
       can actually pay out, independent of how the abuse happens.
    """
    referrer = _supabase_admin.table("users").select(
        "id, email, subscription_status, stripe_subscription_id, subscription_current_period_end, stripe_card_fingerprint"
    ).eq("referral_code", referred_by_code).execute()
    if not referrer.data:
        return  # code didn't match anyone—nothing to reward, fail quietly
    ref_row = referrer.data[0]

    if card_fingerprint and referred_user_id:
        _supabase_admin.table("users").update({"stripe_card_fingerprint": card_fingerprint}).eq("id", referred_user_id).execute()

        if ref_row.get("stripe_card_fingerprint") and card_fingerprint == ref_row["stripe_card_fingerprint"]:
            _supabase_admin.table("referral_fraud_blocks").insert({
                "referrer_email": ref_row.get("email", ""), "referred_email": referred_email or "",
                "reason": "same_card_as_referrer",
            }).execute()
            print(f"[referral] reward blocked—referred account used the same card as the referrer ({ref_row.get('email')})")
            return

    thirty_days_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
    recent_rewards = _supabase_admin.table("admin_grant_log").select("id").eq("target_email", ref_row.get("email", "")) \
        .eq("action", "referral_reward").gte("created_at", thirty_days_ago).execute()
    if len(recent_rewards.data or []) >= _REWARD_CAP_PER_30_DAYS:
        _supabase_admin.table("referral_fraud_blocks").insert({
            "referrer_email": ref_row.get("email", ""), "referred_email": referred_email or "",
            "reason": "reward_cap_reached",
        }).execute()
        print(f"[referral] reward blocked—{ref_row.get('email')} already hit the {_REWARD_CAP_PER_30_DAYS}-reward cap for the last 30 days")
        return

    _grant_free_month(ref_row["id"], ref_row)
    _supabase_admin.table("admin_grant_log").insert({
        "target_email": ref_row.get("email", ""), "action": "referral_reward", "details": {"referred_email": referred_email},
    }).execute()


@router.get("/referral/my-code")
def get_my_referral_code(authorization: str | None = Header(None)):
    user_id = _verify_user(authorization)
    code = _get_or_create_referral_code(user_id)
    return {"code": code}


@router.post("/billing/create-checkout-session")
def create_checkout_session(req: CreateCheckoutRequest, authorization: str | None = Header(None)):
    user_id = _verify_user(authorization)
    if req.plan not in _PRICE_IDS or not _PRICE_IDS[req.plan]:
        raise HTTPException(status_code=400, detail=f"No price configured for plan '{req.plan}'")
    row = _supabase_admin.table("users").select("email").eq("id", user_id).single().execute()
    customer_id = _get_or_create_stripe_customer(user_id, row.data.get("email") if row.data else None)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": _PRICE_IDS[req.plan], "quantity": 1}],
        success_url=f"{_APP_BASE_URL}/billing?checkout=success",
        cancel_url=f"{_APP_BASE_URL}/billing?checkout=canceled",
        metadata={"supabase_user_id": user_id, "plan": req.plan},
    )
    return {"url": session.url}


@router.post("/billing/create-portal-session")
def create_portal_session(authorization: str | None = Header(None)):
    user_id = _verify_user(authorization)
    row = _supabase_admin.table("users").select("stripe_customer_id").eq("id", user_id).single().execute()
    customer_id = row.data.get("stripe_customer_id") if row.data else None
    if not customer_id:
        raise HTTPException(status_code=400, detail="No billing account found for this user yet")

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=f"{_APP_BASE_URL}/billing",
    )
    return {"url": session.url}


@router.post("/billing/webhook")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(None, alias="stripe-signature")):
    """Called directly by Stripe, never by the frontend—verified via
    Stripe's own request signature instead of a user session, proving
    the payload genuinely came from Stripe."""
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, _WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event["type"]
    data = event["data"]["object"]

    def _update_by_customer(customer_id: str, fields: dict):
        _supabase_admin.table("users").update(fields).eq("stripe_customer_id", customer_id).execute()

    def _status_from_stripe_sub_status(stripe_status: str, cancel_at_period_end: bool) -> str:
        if stripe_status == "trialing":
            return "trial"
        if stripe_status == "active" and cancel_at_period_end:
            return "canceling"  # still has access, but won't renew
        if stripe_status == "active":
            return "active"
        if stripe_status == "past_due":
            return "past_due"
        return "free"

    if event_type == "checkout.session.completed":
        plan = (data.get("metadata") or {}).get("plan", "monthly")
        sub_id = data.get("subscription")
        supabase_user_id = (data.get("metadata") or {}).get("supabase_user_id")
        # A checkout with a trial completes into Stripe status
        # "trialing", not "active"—fetching the real subscription is
        # what tells the two apart, since the checkout session itself
        # doesn't carry that distinction directly.
        status = "active"
        period_end_ts = None
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)
            status = _status_from_stripe_sub_status(sub.get("status"), sub.get("cancel_at_period_end", False))
            period_end_ts = sub.get("current_period_end") or sub.get("trial_end")
        fields = {"subscription_status": status, "stripe_subscription_id": sub_id, "subscription_plan": plan}
        if period_end_ts:
            fields["subscription_current_period_end"] = datetime.datetime.fromtimestamp(period_end_ts, tz=datetime.timezone.utc).isoformat()
        _update_by_customer(data["customer"], fields)

        # Referral reward—fires once, the moment this person's own
        # subscription actually starts, not at signup. Guarded by
        # referral_reward_granted so a later renewal webhook (which
        # also fires checkout-adjacent events in some flows) can never
        # reward the same referrer twice for the same person.
        if supabase_user_id:
            user_row = _supabase_admin.table("users").select("email, referred_by_code, referral_reward_granted").eq("id", supabase_user_id).single().execute()
            if user_row.data and user_row.data.get("referred_by_code") and not user_row.data.get("referral_reward_granted"):
                try:
                    card_fingerprint = _get_card_fingerprint(sub_id) if sub_id else None
                    _reward_referrer(
                        user_row.data["referred_by_code"],
                        referred_user_id=supabase_user_id,
                        referred_email=user_row.data.get("email"),
                        card_fingerprint=card_fingerprint,
                    )
                    _supabase_admin.table("users").update({"referral_reward_granted": True}).eq("id", supabase_user_id).execute()
                except Exception as e:
                    print(f"[referral] reward failed for referrer of {supabase_user_id} (their own subscription still started fine): {e}")

    elif event_type == "customer.subscription.updated":
        sub = data
        status = _status_from_stripe_sub_status(sub.get("status"), sub.get("cancel_at_period_end", False))
        period_end = sub.get("current_period_end") or sub.get("trial_end")
        fields = {"subscription_status": status}
        if period_end:
            fields["subscription_current_period_end"] = datetime.datetime.fromtimestamp(period_end, tz=datetime.timezone.utc).isoformat()
        _update_by_customer(sub["customer"], fields)

    elif event_type == "customer.subscription.deleted":
        # Covers both a real cancellation AND an admin "cancel trial,
        # no charge" action, since both end the Stripe subscription the
        # same way—this webhook is what actually flips access off,
        # not the admin endpoint directly, so it stays correct even if
        # Stripe's own dashboard is used to cancel something manually.
        _update_by_customer(data["customer"], {
            "subscription_status": "free",
            "stripe_subscription_id": None,
            "subscription_current_period_end": None,
        })

    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer")
        if customer_id:
            _update_by_customer(customer_id, {"subscription_status": "past_due"})

    return {"received": True}


# ---------------------------------------------------------------------
# Admin endpoints—every single one below requires _verify_admin,
# which means a valid session AND is_admin = true on that account.
# ---------------------------------------------------------------------

class LookupUserRequest(BaseModel):
    email: str


@router.post("/admin/lookup-user")
def lookup_user(req: LookupUserRequest, authorization: str | None = Header(None)):
    """Real current state before you act on it—built specifically so
    granting something can't silently overwrite a status you didn't
    realize was already there."""
    _verify_admin(authorization)
    row = _supabase_admin.table("users").select(
        "id, email, subscription_status, subscription_plan, subscription_current_period_end, stripe_customer_id"
    ).eq("email", req.email).single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")

    period_end = row.data.get("subscription_current_period_end")
    days_remaining = None
    if period_end:
        end_dt = datetime.datetime.fromisoformat(period_end.replace("Z", "+00:00"))
        days_remaining = max(0, (end_dt - datetime.datetime.now(datetime.timezone.utc)).days)

    is_banned = False
    try:
        auth_user = _supabase_admin.auth.admin.get_user_by_id(row.data["id"])
        is_banned = bool(auth_user.user.banned_until) if auth_user and auth_user.user else False
    except Exception:
        pass  # if this lookup fails, default to "not banned" rather than blocking the whole response

    history = _supabase_admin.table("admin_grant_log").select("action, details, created_at") \
        .eq("target_email", req.email).order("created_at", desc=True).limit(10).execute()

    return {
        "email": row.data["email"],
        "subscription_status": row.data.get("subscription_status"),
        "subscription_plan": row.data.get("subscription_plan"),
        "subscription_current_period_end": period_end,
        "days_remaining": days_remaining,
        "is_banned": is_banned,
        "has_stripe_customer": bool(row.data.get("stripe_customer_id")),
        "history": history.data or [],
    }


@router.get("/admin/analytics")
def get_analytics(authorization: str | None = Header(None)):
    """Aggregate usage numbers pulled from data already being collected
    for other reasons—billing status, saved readings, chart
    creation, referral activity—not a new tracking system. Nothing
    here is sold or shared outside the app; it's for deciding what to
    build or fix next.
    """
    _verify_admin(authorization)

    # Counted in Python rather than via a SQL GROUP BY—Supabase's
    # client doesn't expose grouped aggregates directly, and at the
    # scale this app is actually at, fetching one column for every
    # user and counting it here is simpler and completely fine. Worth
    # revisiting with a real SQL aggregate if the user base gets large
    # enough that fetching every row becomes wasteful.
    users = _supabase_admin.table("users").select("subscription_status, created_at").execute()
    status_counts: dict[str, int] = {}
    for row in users.data or []:
        status = row.get("subscription_status") or "free"
        status_counts[status] = status_counts.get(status, 0) + 1
    total_users = len(users.data or [])

    now = datetime.datetime.now(datetime.timezone.utc)
    signups_7d = 0
    signups_30d = 0
    for row in users.data or []:
        created = row.get("created_at")
        if not created:
            continue
        created_dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
        age_days = (now - created_dt).days
        if age_days <= 7:
            signups_7d += 1
        if age_days <= 30:
            signups_30d += 1

    paid_statuses = {"active", "trial", "lifetime", "canceling", "past_due"}
    paid_count = sum(count for status, count in status_counts.items() if status in paid_statuses)
    conversion_rate = round(paid_count / total_users * 100, 1) if total_users else 0.0

    charts = _supabase_admin.table("charts").select("chart_type").execute()
    chart_type_counts: dict[str, int] = {}
    for row in charts.data or []:
        chart_type = row.get("chart_type") or "unknown"
        chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1

    saved = _supabase_admin.table("saved_readings").select("surface").execute()
    saved_by_surface: dict[str, int] = {}
    for row in saved.data or []:
        surface = row.get("surface") or "unknown"
        saved_by_surface[surface] = saved_by_surface.get(surface, 0) + 1

    referral_rewards = _supabase_admin.table("admin_grant_log").select("id", count="exact").eq("action", "referral_reward").execute()
    fraud_blocks = _supabase_admin.table("referral_fraud_blocks").select("id", count="exact").execute()
    push_subs = _supabase_admin.table("push_subscriptions").select("user_id", count="exact").execute()

    return {
        "total_users": total_users,
        "status_counts": status_counts,
        "signups_7d": signups_7d,
        "signups_30d": signups_30d,
        "conversion_rate": conversion_rate,
        "chart_type_counts": chart_type_counts,
        "saved_readings_by_surface": saved_by_surface,
        "referral_rewards_granted": referral_rewards.count or 0,
        "referral_fraud_blocked": fraud_blocks.count or 0,
        "push_subscribers": push_subs.count or 0,
    }


@router.get("/admin/users")
def list_users(status: str | None = None, offset: int = 0, limit: int = 50, authorization: str | None = Header(None)):
    """Browses users—filtered by status if given, or everyone (most
    recent signups first) if not. Paginated with offset/limit rather
    than a flat cap, since a growing user base needs an actual way to
    see more than the first page, not just a higher ceiling."""
    _verify_admin(authorization)
    limit = max(1, min(limit, 100))  # never return an unbounded page, and never accept a nonsense value
    query = _supabase_admin.table("users").select(
        "email, subscription_status, subscription_plan, subscription_current_period_end, created_at",
        count="exact",
    )
    if status:
        query = query.eq("subscription_status", status)
    result = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return {"users": result.data or [], "total": result.count, "offset": offset, "limit": limit}


class GrantAccessRequest(BaseModel):
    email: str
    grant: str  # "trial" | "lifetime" | "reset_to_free"
    trial_months: int | None = None  # required when grant == "trial"
    trial_plan: str | None = "monthly"  # which plan the trial converts into


@router.post("/admin/grant-access")
def grant_access(req: GrantAccessRequest, authorization: str | None = Header(None)):
    _verify_admin(authorization)

    target = _supabase_admin.table("users").select("id, email").eq("email", req.email).single().execute()
    if not target.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")
    target_id = target.data["id"]

    if req.grant == "lifetime":
        # Cancel any real, currently-billing Stripe subscription first,
        # same as reset_to_free already does—a real bug otherwise:
        # upgrading someone to lifetime while their old subscription ID
        # stays on their row means a LATER referral reward (which only
        # checked for a subscription ID, not the actual status) could
        # modify that stale subscription—either leaving them
        # double-billed alongside lifetime access, or crashing on a
        # subscription that no longer exists. Clearing it here removes
        # the stale data at its source, and _reward_referrer below now
        # also checks subscription_status directly as a second,
        # independent safeguard.
        row = _supabase_admin.table("users").select("stripe_subscription_id").eq("id", target_id).single().execute()
        sub_id = row.data.get("stripe_subscription_id") if row.data else None
        if sub_id:
            try:
                stripe.Subscription.delete(sub_id)
            except stripe.error.InvalidRequestError:
                pass
        _supabase_admin.table("users").update({
            "subscription_status": "lifetime", "subscription_current_period_end": None, "stripe_subscription_id": None,
        }).eq("id", target_id).execute()
        _log_admin_action(req.email, "grant_lifetime")
        return {"success": True, "email": req.email, "grant": "lifetime"}

    elif req.grant == "trial":
        if not req.trial_months or req.trial_months not in (1, 3, 6, 12):
            raise HTTPException(status_code=400, detail="trial_months must be 1, 3, 6, or 12")
        plan = req.trial_plan if req.trial_plan in _PRICE_IDS else "monthly"
        if not _PRICE_IDS[plan]:
            raise HTTPException(status_code=400, detail=f"No price configured for plan '{plan}'")

        # A real Stripe subscription with trial_period_days set—this
        # is what makes automatic billing at the end of the trial and
        # cancel-during-trial-through-the-normal-portal both work for
        # free, using Stripe's own native trial handling, rather than
        # a manually-set date with no card behind it that nothing could
        # ever actually charge.
        customer_id = _get_or_create_stripe_customer(target_id, req.email)
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": _PRICE_IDS[plan], "quantity": 1}],
            subscription_data={"trial_period_days": 30 * req.trial_months},
            success_url=f"{_APP_BASE_URL}/billing?checkout=success",
            cancel_url=f"{_APP_BASE_URL}/billing?checkout=canceled",
            metadata={"supabase_user_id": target_id, "plan": plan},
        )
        _log_admin_action(req.email, "grant_trial", {"months": req.trial_months, "plan": plan})
        # The actual trial doesn't start until this link is opened and
        # a card is entered—Stripe requires a payment method to
        # create the subscription at all, even one that won't be
        # charged until the trial ends. This URL is what needs to
        # actually reach the person—send it to them however you
        # normally would (email, text, DM).
        return {"success": True, "email": req.email, "grant": "trial", "checkout_url": session.url}

    elif req.grant == "reset_to_free":
        # If there's a real Stripe subscription behind their current
        # status, cancel it too—resetting the database status alone
        # while Stripe keeps billing them would be a real, bad bug.
        row = _supabase_admin.table("users").select("stripe_subscription_id").eq("id", target_id).single().execute()
        sub_id = row.data.get("stripe_subscription_id") if row.data else None
        if sub_id:
            try:
                stripe.Subscription.delete(sub_id)
            except stripe.error.InvalidRequestError:
                pass  # already canceled or doesn't exist—fine, proceed to reset our own record regardless
        _supabase_admin.table("users").update({
            "subscription_status": "free", "subscription_current_period_end": None, "stripe_subscription_id": None,
        }).eq("id", target_id).execute()
        _log_admin_action(req.email, "reset_to_free")
        return {"success": True, "email": req.email, "grant": "reset_to_free"}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown grant type: {req.grant}")


class EndTrialRequest(BaseModel):
    email: str
    mode: str  # "bill_now" | "cancel_no_charge"


@router.post("/admin/end-trial")
def end_trial(req: EndTrialRequest, authorization: str | None = Header(None)):
    _verify_admin(authorization)
    row = _supabase_admin.table("users").select("stripe_subscription_id, subscription_status").eq("email", req.email).single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")
    if row.data.get("subscription_status") != "trial":
        raise HTTPException(status_code=400, detail=f"{req.email} isn't currently on a trial")
    sub_id = row.data.get("stripe_subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="No Stripe subscription found for this trial")

    if req.mode == "bill_now":
        # Ends the trial immediately—Stripe attempts to charge the
        # card on file right away and the subscription becomes active.
        stripe.Subscription.modify(sub_id, trial_end="now", proration_behavior="none")
        _log_admin_action(req.email, "end_trial_bill_now")
    elif req.mode == "cancel_no_charge":
        # Ends the subscription outright—no charge, access ends via
        # the same webhook a real cancellation goes through.
        stripe.Subscription.delete(sub_id)
        _log_admin_action(req.email, "cancel_trial_no_charge")
    else:
        raise HTTPException(status_code=400, detail="mode must be 'bill_now' or 'cancel_no_charge'")

    return {"success": True, "email": req.email, "mode": req.mode}


class BanUserRequest(BaseModel):
    email: str


@router.post("/admin/ban-user")
def ban_user(req: BanUserRequest, authorization: str | None = Header(None)):
    _verify_admin(authorization)
    row = _supabase_admin.table("users").select("id").eq("email", req.email).single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")
    # A ban blocks future logins and session refreshes at Supabase's own
    # auth layer—not a custom flag this app has to remember to check
    # everywhere. An access token issued just before the ban can keep
    # working until it naturally expires (short-lived, typically under
    # an hour) since Supabase doesn't retroactively revoke tokens
    # already issued; the ban prevents getting a new one.
    _supabase_admin.auth.admin.update_user_by_id(row.data["id"], {"ban_duration": "876000h"})  # ~100 years
    _log_admin_action(req.email, "ban")
    return {"success": True, "email": req.email}


@router.post("/admin/unban-user")
def unban_user(req: BanUserRequest, authorization: str | None = Header(None)):
    _verify_admin(authorization)
    row = _supabase_admin.table("users").select("id").eq("email", req.email).single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")
    _supabase_admin.auth.admin.update_user_by_id(row.data["id"], {"ban_duration": "none"})
    _log_admin_action(req.email, "unban")
    return {"success": True, "email": req.email}


class DeleteUserRequest(BaseModel):
    email: str


@router.post("/admin/delete-user")
def delete_user(req: DeleteUserRequest, authorization: str | None = Header(None)):
    _verify_admin(authorization)
    row = _supabase_admin.table("users").select("id, stripe_subscription_id").eq("email", req.email).single().execute()
    if not row.data:
        raise HTTPException(status_code=404, detail=f"No user found with email {req.email}")
    user_id = row.data["id"]

    # Cancel any real Stripe subscription first—deleting the account
    # should not leave someone getting charged for access that no
    # longer exists for them to use.
    sub_id = row.data.get("stripe_subscription_id")
    if sub_id:
        try:
            stripe.Subscription.delete(sub_id)
        except stripe.error.InvalidRequestError:
            pass

    # Delete the application-data row explicitly rather than assuming a
    # cascade constraint handles it—safer not to rely on a foreign
    # key behavior that was never directly confirmed.
    _supabase_admin.table("users").delete().eq("id", user_id).execute()
    _supabase_admin.auth.admin.delete_user(user_id)
    _log_admin_action(req.email, "delete_account")
    return {"success": True, "email": req.email}
