import http.client
import json
import secrets
import threading
import unittest
from http.cookies import SimpleCookie
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dashboard.admin_auth import COOKIE_NAME, password_hash
from dashboard.server import DashboardServer


class DashboardAuthTests(unittest.TestCase):
    def write_credentials(self, path, password):
        salt = secrets.token_bytes(16)
        path.write_text(json.dumps({
            "salt": salt.hex(), "verifier": password_hash(password, salt).hex(),
        }), encoding="utf-8")

    def start_server(self, credentials):
        server = DashboardServer(("127.0.0.1", 0), credentials=credentials)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join, 5)
        self.addCleanup(server.shutdown)
        self.port = server.server_port
        self.origin = f"http://127.0.0.1:{self.port}"

    def request(self, method, path, payload=None, cookie=None, dashboard_header=True):
        headers = {}
        if payload is not None:
            headers.update({"Content-Type": "application/json", "Origin": self.origin})
            if dashboard_header:
                headers["X-Dashboard-Request"] = "1"
        if cookie is not None:
            headers["Cookie"] = cookie
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            client.request(method, path, json.dumps(payload) if payload is not None else None, headers)
            response = client.getresponse()
            return response.status, response.headers, response.read()
        finally:
            client.close()

    def test_guest_chat_and_admin_session_lifecycle(self):
        directory = self.enterContext(TemporaryDirectory())
        credentials = Path(directory) / "admin.json"
        password = secrets.token_urlsafe(24)
        self.write_credentials(credentials, password)
        self.start_server(credentials)

        self.assertEqual(self.request("GET", "/")[0], 200)
        for path in ("/api/git", "/api/docs", "/api/tasks", "/api/jobs/example", "/api/storage"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 401)
        for path in ("/api/git/pull", "/api/scripts/run", "/api/tasks/benchmark", "/api/storage/sync"):
            with self.subTest(path=path):
                self.assertEqual(self.request("POST", path, {})[0], 401)

        job_id = "a" * 32
        job = {"id": job_id, "status": "queued"}
        with patch("dashboard.server.recommendations") as recommendations:
            recommendations.submit.return_value = job
            recommendations.get.return_value = job
            status, _, data = self.request("POST", "/api/cpu-recommend", {"query": "graph learning"})
            self.assertEqual(status, 202)
            self.assertEqual(json.loads(data), job)
            self.assertEqual(self.request("GET", f"/api/cpu-recommend/{job_id}")[0], 200)

        login = {"username": "admin", "password": password}
        self.assertEqual(
            self.request("POST", "/api/auth/login", login, dashboard_header=False)[0], 403,
        )
        bad_login = dict(login, password=secrets.token_urlsafe(24))
        self.assertEqual(self.request("POST", "/api/auth/login", bad_login)[0], 401)
        status, headers, _ = self.request("POST", "/api/auth/login", login)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        cookie = SimpleCookie(headers["Set-Cookie"])[COOKIE_NAME]
        self.assertTrue(cookie["secure"] and cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Strict")
        session = f"{COOKIE_NAME}={cookie.value}"
        self.assertEqual(json.loads(self.request("GET", "/api/auth/session", cookie=session)[2]),
                         {"authenticated": True})
        self.assertEqual(self.request("GET", "/api/docs", cookie=session)[0], 200)
        self.assertEqual(
            self.request("POST", "/api/scripts/run", {}, session, dashboard_header=False)[0], 403,
        )
        self.assertEqual(self.request("POST", "/api/auth/logout", {}, session)[0], 200)
        self.assertEqual(self.request("GET", "/api/docs", cookie=session)[0], 401)

    def test_admin_credentials_can_be_created_and_rotated_on_running_server(self):
        directory = self.enterContext(TemporaryDirectory())
        credentials = Path(directory) / "admin.json"
        self.start_server(credentials)
        self.assertEqual(self.request("POST", "/api/auth/login", {})[0], 503)
        self.assertEqual(self.request("GET", "/api/docs")[0], 401)
        password = secrets.token_urlsafe(24)
        self.write_credentials(credentials, password)
        login = {"username": "admin", "password": password}
        status, headers, _ = self.request("POST", "/api/auth/login", login)
        self.assertEqual(status, 200)
        cookie = SimpleCookie(headers["Set-Cookie"])[COOKIE_NAME]
        session = f"{COOKIE_NAME}={cookie.value}"
        self.write_credentials(credentials, secrets.token_urlsafe(24))
        session_response = self.request("GET", "/api/auth/session", cookie=session)
        self.assertEqual(json.loads(session_response[2]), {"authenticated": False})
