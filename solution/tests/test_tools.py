"""Rubric: support tools with database abstraction, validation and error handling."""
import pytest

from agentic.db import cultpass, cultpass_engine, session_scope
from agentic.tools import cultpass_ops as ops
from agentic.tools.registry import ToolRegistry, llm_tools


def _exp(title):
    with session_scope(cultpass_engine()) as s:
        e = s.query(cultpass.Experience).filter_by(title=title).one()
        return e.experience_id, e.slots_available


def _res(user_id, title):
    listing = ops.list_reservations(user_id)["data"]
    return next(r for r in listing if r["title"] == title)


def test_profile_ok_and_structured(data_env):
    r = ops.get_customer_profile("f556c0")
    assert r["ok"] is True
    d = r["data"]
    assert d["full_name"] == "Bob Stone" and d["subscription"]["tier"] == "basic"
    assert d["quota"] == {"monthly_quota": 4, "used_this_cycle": 2, "remaining": 2,
                          "cycle_resets_at": d["quota"]["cycle_resets_at"]}


@pytest.mark.parametrize("bad, code", [("nope", "not_found"), ("", "invalid_argument"), (None, "invalid_argument"),
                                       ("x" * 100, "invalid_argument")])
def test_profile_validation(data_env, bad, code):
    r = ops.get_customer_profile(bad)
    assert r["ok"] is False and r["error"]["code"] == code


def test_list_and_search(data_env):
    res = ops.list_reservations("f556c0")["data"]
    assert [r["title"] for r in res] == ["Carnival History Tour in Olinda", "Samba Night at Lapa"]
    assert res[0]["free_cancellation"] is False and res[1]["free_cancellation"] is True
    assert ops.list_reservations("88382b", status="cancelled")["data"][0]["title"] == "Sunset Paddleboarding"
    found = ops.search_experiences(query="samba")["data"]
    assert [e["title"] for e in found] == ["Samba Night at Lapa"]
    assert all("Rio" in e["location"] for e in ops.search_experiences(location="Rio")["data"])
    assert "Ibirapuera Park Bike Ride" not in [e["title"] for e in ops.search_experiences(only_available=True)["data"]]
    assert ops.search_experiences(limit=0)["error"]["code"] == "invalid_argument"


def test_reserve_success_decrements_slots_and_quota(data_env):
    eid, slots = _exp("Pelourinho Colonial Walk")  # premium experience
    r = ops.reserve_experience("f556c0", eid)
    assert r["ok"], r
    assert r["data"]["monthly_quota_remaining_after_booking"] == 1 and r["data"]["premium_fee_applies"] is True
    assert _exp("Pelourinho Colonial Walk")[1] == slots - 1
    assert ops.reserve_experience("f556c0", eid)["error"]["code"] == "conflict"  # already reserved


@pytest.mark.parametrize("user, title, code", [
    ("f556c0", "Ibirapuera Park Bike Ride", "sold_out"),
    ("a4ab87", "Samba Night at Lapa", "blocked"),
    ("f1f10d", "Samba Night at Lapa", "forbidden"),       # paused subscription
    ("f556c0", "does-not-exist", "not_found"),
])
def test_reserve_rules(data_env, user, title, code):
    eid = _exp(title)[0] if title != "does-not-exist" else "zzzzzz"
    assert ops.reserve_experience(user, eid)["error"]["code"] == code


def test_reserve_quota_exceeded(data_env):
    ops.reserve_experience("f556c0", _exp("Modern Art at MASP")[0])
    ops.reserve_experience("f556c0", _exp("Christ the Redeemer Experience")[0])
    r = ops.reserve_experience("f556c0", _exp("Pelourinho Colonial Walk")[0])
    assert r["error"]["code"] == "quota_exceeded" and "resets_at" in r["error"]


def test_cancel_free_and_late(data_env):
    samba = _res("f556c0", "Samba Night at Lapa")
    _, slots = _exp("Samba Night at Lapa")
    r = ops.cancel_reservation("f556c0", samba["reservation_id"])
    assert r["data"]["credit_returned"] is True and r["data"]["status"] == "cancelled"
    assert _exp("Samba Night at Lapa")[1] == slots + 1
    late = ops.cancel_reservation("f556c0", _res("f556c0", "Carnival History Tour in Olinda")["reservation_id"])
    assert late["data"]["credit_returned"] is False and late["data"]["status"] == "cancelled_late"
    again = ops.cancel_reservation("f556c0", samba["reservation_id"])
    assert again["error"]["code"] == "conflict"


def test_cannot_cancel_someone_elses_reservation(data_env):
    cathy = _res("88382b", "Modern Art at MASP")
    r = ops.cancel_reservation("f556c0", cathy["reservation_id"])
    assert r["error"]["code"] == "not_found"  # does not reveal it exists


def test_subscription_lifecycle(data_env):
    assert ops.cancel_subscription("f556c0")["error"]["code"] == "confirmation_required"
    assert ops.pause_subscription("f556c0")["data"]["status"] == "paused"
    assert ops.pause_subscription("f556c0")["error"]["code"] == "conflict"
    assert ops.resume_subscription("f556c0")["data"]["status"] == "active"
    c = ops.cancel_subscription("f556c0", customer_confirmed=True)["data"]
    assert c["status"] == "cancelled" and c["effective_until"]
    assert ops.pause_subscription("a4ab87")["error"]["code"] == "blocked"


def test_refund_request_validation_and_dedup(data_env):
    assert ops.submit_refund_request("88382b", "")["error"]["code"] == "invalid_argument"
    assert ops.submit_refund_request("88382b", "dup", amount=-5)["error"]["code"] == "invalid_argument"
    assert ops.submit_refund_request("88382b", "dup", amount="abc")["error"]["code"] == "invalid_argument"
    first = ops.submit_refund_request("88382b", "charged twice", amount=29.9)["data"]
    assert first["status"] == "pending_approval" and first["duplicate"] is False
    second = ops.submit_refund_request("88382b", "charged twice again")["data"]
    assert second["duplicate"] is True and second["action_id"] == first["action_id"]


def test_db_missing_returns_db_error(data_env, tmp_path, monkeypatch):
    monkeypatch.setenv("UDAHUB_DATA_DIR", str(tmp_path / "nowhere"))
    r = ops.get_customer_profile("f556c0")
    assert r == {"ok": False, "error": {"code": "db_error", "message": r["error"]["message"]}}
    assert "01_external_db_setup" in r["error"]["message"]


def test_registry_injects_identity_and_validates(data_env):
    reg = ToolRegistry("local")
    r = reg.call("get_customer_profile", {"user_id": "a4ab87"}, user_id="f556c0")  # LLM-supplied id ignored
    assert r["data"]["user_id"] == "f556c0" and r["_meta"]["transport"] == "local"
    assert reg.call("drop_tables", {}, user_id="f556c0")["error"]["code"] == "invalid_argument"
    assert reg.call("list_reservations", {"bogus": 1}, user_id="f556c0")["error"]["code"] == "invalid_argument"
    assert reg.call("list_reservations", {}, user_id=None)["error"]["code"] == "forbidden"
    refund = reg.call("submit_refund_request", {"reason": "x"}, user_id="88382b", ticket_id="t-1")
    assert refund["ok"]


def test_llm_tool_specs_hide_identity_arguments():
    specs = {t.name: t for t in llm_tools(list(ops.OPERATIONS))}
    assert set(specs) == set(ops.OPERATIONS)
    for spec in specs.values():
        props = spec.args_schema.model_json_schema().get("properties", {})
        assert "user_id" not in props and "ticket_id" not in props
        assert spec.description
    assert "customer_confirmed" in specs["cancel_subscription"].args_schema.model_json_schema()["properties"]
