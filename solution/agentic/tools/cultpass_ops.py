"""CultPass support operations: the database abstraction behind every tool.

Each function is a plain, synchronous Python function that:

* validates its arguments,
* talks to the CultPass database (or UDA-Hub for refund requests) through
  SQLAlchemy only - agents never see SQL or table layouts,
* returns a structured envelope: ``{"ok": True, "data": ...}`` or
  ``{"ok": False, "error": {"code": ..., "message": ...}}``, and
* never raises for expected failures (unknown user, sold-out event, ...).

The same functions are exposed to agents two ways (see tools/registry.py):
through the FastMCP server in tools/mcp_server.py (default) and as in-process
LangChain tools (fallback / fast tests).
"""
from __future__ import annotations

import functools
import inspect
import json
import uuid
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from agentic.db import (DatabaseNotInitialised, core_engine, cultpass, cultpass_engine,
                        session_scope, udahub)

ERROR_CODES = ("invalid_argument", "not_found", "blocked", "forbidden", "conflict",
               "quota_exceeded", "sold_out", "confirmation_required", "db_error")

CANCELLATION_WINDOW_HOURS = 24


class OpError(Exception):
    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


def ok(data: Any) -> dict:
    return {"ok": True, "data": data}


def fail(code: str, message: str, **extra) -> dict:
    return {"ok": False, "error": {"code": code, "message": message, **extra}}


def operation(fn: Callable) -> Callable:
    """Turn exceptions into the error envelope so callers get data, not tracebacks."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return ok(fn(*args, **kwargs))
        except OpError as e:
            return fail(e.code, e.message, **e.extra)
        except DatabaseNotInitialised as e:
            return fail("db_error", str(e))
        except SQLAlchemyError as e:
            return fail("db_error", f"database error: {e.__class__.__name__}")
    # The wrapper returns the envelope dict whatever the inner function returns;
    # say so explicitly, or FastMCP would publish the inner return type as the
    # tool's output schema and reject the envelope.
    wrapper.__annotations__ = {**fn.__annotations__, "return": "dict"}
    wrapper.__signature__ = inspect.signature(fn).replace(return_annotation=dict)
    del wrapper.__wrapped__
    return wrapper


def _require_str(name: str, value: Any, max_len: int = 64) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpError("invalid_argument", f"'{name}' must be a non-empty string")
    value = value.strip()
    if len(value) > max_len:
        raise OpError("invalid_argument", f"'{name}' is too long")
    return value


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="minutes") if dt else None


def _get_user(s, user_id: str) -> cultpass.User:
    user = s.get(cultpass.User, _require_str("user_id", user_id))
    if user is None:
        raise OpError("not_found", f"no CultPass user with id '{user_id}'")
    return user


def _cycle_start(started_at: datetime, now: datetime) -> datetime:
    """Most recent monthly anniversary of the subscription start (quota reset)."""
    day = min(started_at.day, 28)
    start = now.replace(day=day, hour=started_at.hour, minute=started_at.minute, second=0, microsecond=0)
    if start > now:
        month, year = (now.month - 1, now.year) if now.month > 1 else (12, now.year - 1)
        start = start.replace(year=year, month=month)
    return start


def _next_cycle(start: datetime) -> datetime:
    month, year = (start.month + 1, start.year) if start.month < 12 else (1, start.year + 1)
    return start.replace(year=year, month=month)


def _quota(s, user: cultpass.User, now: datetime) -> dict:
    sub = user.subscription
    if sub is None:
        return {"monthly_quota": 0, "used_this_cycle": 0, "remaining": 0, "cycle_resets_at": None}
    start = _cycle_start(sub.started_at, now)
    used = (s.query(cultpass.Reservation)
            .filter(cultpass.Reservation.user_id == user.user_id,
                    cultpass.Reservation.status.in_(["reserved", "attended", "no_show", "cancelled_late"]),
                    cultpass.Reservation.created_at >= start)
            .count())
    return {"monthly_quota": sub.monthly_quota, "used_this_cycle": used,
            "remaining": max(sub.monthly_quota - used, 0), "cycle_resets_at": _iso(_next_cycle(start))}


def _reservation_dict(r: cultpass.Reservation, now: datetime) -> dict:
    exp = r.experience
    hours = round((exp.when - now).total_seconds() / 3600, 1) if exp and exp.when else None
    return {
        "reservation_id": r.reservation_id, "status": r.status,
        "experience_id": r.experience_id, "title": exp.title if exp else None,
        "location": exp.location if exp else None, "when": _iso(exp.when) if exp else None,
        "is_premium": exp.is_premium if exp else None, "hours_until_event": hours,
        "free_cancellation": hours is not None and hours > CANCELLATION_WINDOW_HOURS,
    }


def _experience_dict(e: cultpass.Experience) -> dict:
    return {"experience_id": e.experience_id, "title": e.title, "description": e.description,
            "location": e.location, "when": _iso(e.when), "slots_available": e.slots_available,
            "is_premium": e.is_premium}


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

@operation
def get_customer_profile(user_id: str) -> dict:
    """Look up a CultPass customer: blocked flag, subscription (status, tier, monthly quota) and quota used this billing cycle."""
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        sub = user.subscription
        active = [r for r in user.reservations if r.status == "reserved"]
        return {
            "user_id": user.user_id, "full_name": user.full_name, "email": user.email,
            "is_blocked": bool(user.is_blocked), "member_since": _iso(user.created_at),
            "subscription": None if sub is None else {
                "subscription_id": sub.subscription_id, "status": sub.status, "tier": sub.tier,
                "monthly_quota": sub.monthly_quota, "started_at": _iso(sub.started_at),
                "ended_at": _iso(sub.ended_at),
            },
            "quota": _quota(s, user, now),
            "upcoming_reservations": len(active),
        }


@operation
def list_reservations(user_id: str, status: str | None = None) -> list[dict]:
    """List a customer's reservations (soonest first) with hours until the event and whether cancellation is still free. Optional status filter: reserved, cancelled, cancelled_late."""
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        rows = [r for r in user.reservations if status is None or r.status == status]
        rows.sort(key=lambda r: r.experience.when if r.experience else now)
        return [_reservation_dict(r, now) for r in rows]


@operation
def search_experiences(query: str | None = None, location: str | None = None,
                       only_available: bool = False, premium: bool | None = None,
                       limit: int = 10) -> list[dict]:
    """Search upcoming CultPass experiences by keywords and/or location (state or city). Use only_available=true to hide sold-out ones."""
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise OpError("invalid_argument", "'limit' must be an integer between 1 and 50")
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        q = s.query(cultpass.Experience).filter(cultpass.Experience.when >= now)
        if query:
            terms = [t for t in query.split() if len(t) > 2] or [query]
            q = q.filter(or_(*[or_(cultpass.Experience.title.ilike(f"%{t}%"),
                                   cultpass.Experience.description.ilike(f"%{t}%")) for t in terms]))
        if location:
            q = q.filter(cultpass.Experience.location.ilike(f"%{location}%"))
        if only_available:
            q = q.filter(cultpass.Experience.slots_available > 0)
        if premium is not None:
            q = q.filter(cultpass.Experience.is_premium == bool(premium))
        return [_experience_dict(e) for e in q.order_by(cultpass.Experience.when).limit(limit).all()]


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

@operation
def reserve_experience(user_id: str, experience_id: str) -> dict:
    """Reserve one slot of an experience for the customer. Enforces blocked accounts, active subscription, monthly quota and availability."""
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        if user.is_blocked:
            raise OpError("blocked", "the account is blocked; reservations are disabled")
        sub = user.subscription
        if sub is None or sub.status != "active":
            raise OpError("forbidden", f"subscription is '{sub.status if sub else 'missing'}'; an active plan is required")
        exp = s.get(cultpass.Experience, _require_str("experience_id", experience_id))
        if exp is None:
            raise OpError("not_found", f"no experience with id '{experience_id}'")
        if exp.when < now:
            raise OpError("conflict", "that experience has already taken place")
        if any(r.experience_id == exp.experience_id and r.status == "reserved" for r in user.reservations):
            raise OpError("conflict", "the user already has a reservation for this experience")
        if exp.slots_available <= 0:
            raise OpError("sold_out", "no slots left; the user can join the waitlist in the app")
        quota = _quota(s, user, now)
        if quota["remaining"] <= 0:
            raise OpError("quota_exceeded", "monthly quota used up", resets_at=quota["cycle_resets_at"])
        exp.slots_available -= 1
        res = cultpass.Reservation(reservation_id=uuid.uuid4().hex[:6], user_id=user.user_id,
                                   experience_id=exp.experience_id, status="reserved", created_at=now)
        s.add(res)
        s.flush()
        s.refresh(res)
        out = _reservation_dict(res, now)
        out["premium_fee_applies"] = bool(exp.is_premium and sub.tier != "premium")
        out["monthly_quota_remaining_after_booking"] = quota["remaining"] - 1
        return out


@operation
def cancel_reservation(user_id: str, reservation_id: str) -> dict:
    """Cancel one of the customer's reservations. The credit is returned only if the event is more than 24 hours away."""
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        res = s.get(cultpass.Reservation, _require_str("reservation_id", reservation_id))
        if res is None or res.user_id != user.user_id:  # never reveal other users' reservations
            raise OpError("not_found", f"no reservation '{reservation_id}' for this user")
        if res.status != "reserved":
            raise OpError("conflict", f"reservation is already '{res.status}'")
        info = _reservation_dict(res, now)
        res.status = "cancelled"
        if res.experience:
            res.experience.slots_available += 1
        credit_returned = bool(info["free_cancellation"])
        if not credit_returned:
            # inside the window the credit is consumed: record it so quota maths stays right
            res.status = "cancelled_late"
        info.update(status=res.status, credit_returned=credit_returned)
        return info


def _set_subscription(user_id: str, *, allowed_from: tuple[str, ...], new_status: str,
                      ended_at_cycle_end: bool = False) -> dict:
    now = datetime.now()
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        if user.is_blocked:
            raise OpError("blocked", "the account is blocked; subscription changes need Trust & Safety")
        sub = user.subscription
        if sub is None:
            raise OpError("not_found", "the user has no subscription")
        if sub.status not in allowed_from:
            raise OpError("conflict", f"subscription is '{sub.status}', expected one of {list(allowed_from)}")
        previous = sub.status
        sub.status = new_status
        if ended_at_cycle_end:
            sub.ended_at = _next_cycle(_cycle_start(sub.started_at, now))
        elif new_status == "active":
            sub.ended_at = None
            if previous == "cancelled":
                sub.started_at = now  # reactivation starts a new billing cycle
        return {"subscription_id": sub.subscription_id, "previous_status": previous,
                "status": sub.status, "tier": sub.tier, "effective_until": _iso(sub.ended_at)}


@operation
def pause_subscription(user_id: str) -> dict:
    """Pause the customer's active subscription (billing stops, data is kept)."""
    return _set_subscription(user_id, allowed_from=("active",), new_status="paused")


@operation
def resume_subscription(user_id: str) -> dict:
    """Resume the customer's paused or cancelled subscription."""
    return _set_subscription(user_id, allowed_from=("paused", "cancelled"), new_status="active")


@operation
def cancel_subscription(user_id: str, customer_confirmed: bool = False) -> dict:
    """Cancel the subscription at the end of the billing cycle. Only pass customer_confirmed=true when the customer explicitly asked for or confirmed the cancellation in this conversation."""
    if customer_confirmed is not True:
        raise OpError("confirmation_required",
                      "ask the customer to confirm the cancellation explicitly, then call again with customer_confirmed=true")
    return _set_subscription(user_id, allowed_from=("active", "paused"), new_status="cancelled",
                             ended_at_cycle_end=True)


@operation
def submit_refund_request(user_id: str, reason: str, amount: float | None = None,
                          ticket_id: str | None = None) -> dict:
    """Submit a refund request for support-lead approval. This does NOT refund anything; tell the customer the request is pending review."""
    reason = _require_str("reason", reason, max_len=500)
    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise OpError("invalid_argument", "'amount' must be a number")
        if not 0 < amount <= 1000:
            raise OpError("invalid_argument", "'amount' must be between 0 and 1000")
    with session_scope(cultpass_engine()) as s:
        user = _get_user(s, user_id)
        sub = user.subscription
        snapshot = {"tier": sub.tier if sub else None, "status": sub.status if sub else None}
    action_id = uuid.uuid4().hex[:10]
    with session_scope(core_engine()) as s:
        existing = (s.query(udahub.SupportAction)
                    .filter_by(external_user_id=user_id, action_type="refund_request", status="pending_approval")
                    .first())
        if existing is not None:
            return {"action_id": existing.action_id, "status": existing.status, "duplicate": True,
                    "message": "a refund request is already pending approval for this user"}
        s.add(udahub.SupportAction(
            action_id=action_id, ticket_id=ticket_id, external_user_id=user_id,
            action_type="refund_request", status="pending_approval",
            details=json.dumps({"reason": reason, "amount": amount, "subscription": snapshot}),
        ))
    return {"action_id": action_id, "status": "pending_approval", "duplicate": False,
            "requires": "support lead approval", "sla": "decision by email; 5-10 business days to the card once approved"}


# Name -> function; the registry and the MCP server both build from this.
OPERATIONS: dict[str, Callable[..., dict]] = {
    f.__name__: f for f in (
        get_customer_profile, list_reservations, search_experiences, reserve_experience,
        cancel_reservation, pause_subscription, resume_subscription, cancel_subscription,
        submit_refund_request,
    )
}

# Tools that change data. The workflow logs these with extra care and the QA
# reviewer checks that any action the reply claims actually succeeded.
MUTATING = {"reserve_experience", "cancel_reservation", "pause_subscription",
            "resume_subscription", "cancel_subscription", "submit_refund_request"}
