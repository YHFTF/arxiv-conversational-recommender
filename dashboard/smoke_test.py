"""Small smoke test for a running dashboard server."""

import json
import os
import urllib.error
import urllib.request

BASE = os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:8080")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        assert response.status == 200
        return response.read()


health = json.loads(get("/api/health"))
index = get("/")
assert health["ok"] is True
assert b"app.js" in index and b"admin-form" in index and b"live-rec-form" in index
assert json.loads(get("/api/auth/session")) == {"authenticated": False}
for path in ("/api/git", "/api/docs", "/api/tasks", "/api/scripts", "/api/storage"):
    try:
        get(path)
    except urllib.error.HTTPError as error:
        assert error.code == 401, (path, error.code)
    else:
        raise AssertionError(f"Unauthenticated request was accepted for {path}")
print(f"OK: public dashboard available, administrator routes protected, HTML={len(index)} bytes")
