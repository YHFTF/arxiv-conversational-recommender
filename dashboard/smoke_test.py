"""Small smoke test for a running dashboard server."""

import json
import urllib.request

BASE = "http://127.0.0.1:8080"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        assert response.status == 200
        return response.read()


health = json.loads(get("/api/health"))
git = json.loads(get("/api/git"))
docs = json.loads(get("/api/docs"))
index = get("/")
assert health["ok"] is True
assert "branch" in git and "commits" in git
assert docs["documents"] and docs["tree"]
assert b"app.js" in index and b"styles.css" in index
print(
    f"OK: branch={git['branch']}, docs={len(docs['documents'])}, html={len(index)} bytes"
)
