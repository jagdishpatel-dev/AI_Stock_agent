"""The dashboard API must only answer requests that carry a dashboard session or admin key."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api import server

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "_admin_key", ADMIN_KEY)
    return TestClient(server.app)


def test_data_endpoint_rejects_request_without_session(client: TestClient) -> None:
    res = client.get("/api/daily-pnl")
    assert res.status_code == 401


def test_session_cookie_unlocks_data_endpoints(client: TestClient) -> None:
    session = client.get("/api/session")
    assert session.status_code == 200
    cookie = session.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert "Path=/api" in cookie

    assert client.get("/api/daily-pnl").status_code == 200


def test_tampered_or_expired_session_is_rejected(client: TestClient) -> None:
    client.get("/api/session")
    token = client.cookies["dash_session"]
    expires, _, sig = token.partition(".")

    client.cookies.set("dash_session", f"{int(expires) + 3600}.{sig}", path="/api")
    assert client.get("/api/daily-pnl").status_code == 401

    stale = server._sign_session(int(time.time()) - 1)
    client.cookies.set("dash_session", stale, path="/api")
    assert client.get("/api/daily-pnl").status_code == 401


def test_admin_key_works_without_session(client: TestClient) -> None:
    res = client.get("/api/daily-pnl", headers={"X-Admin-Key": ADMIN_KEY})
    assert res.status_code == 200
    res = client.get("/api/daily-pnl", headers={"X-Admin-Key": "wrong"})
    assert res.status_code == 401


def test_health_and_session_stay_public(client: TestClient) -> None:
    assert client.get("/api/session").status_code == 200
    client.cookies.clear()
    assert client.get("/api/health").status_code == 200


@pytest.mark.parametrize("path", ["/api/logs", "/api/config", "/api/events"])
def test_sensitive_endpoints_need_admin_key_even_with_session(
    client: TestClient, path: str
) -> None:
    client.get("/api/session")
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_disabled(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_cross_origin_requests_get_no_cors_grant(client: TestClient) -> None:
    client.get("/api/session")
    res = client.get("/api/daily-pnl", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in res.headers
