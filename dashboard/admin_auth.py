"""Local administrator credentials and short-lived dashboard sessions."""

import getpass
import hashlib
import json
import secrets
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path


COOKIE_NAME = "__Host-arxiv-admin"
SESSION_SECONDS = 8 * 60 * 60
LOGIN_WINDOW = 60
MAX_FAILED_LOGINS = 5
MAX_SESSIONS = 32
USERNAME = "admin"


def password_hash(password, salt):
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=2**17, r=8, p=1, maxmem=256 * 1024 * 1024, dklen=32,
    )


def set_admin_password(path: Path):
    if not sys.stdin.isatty():
        raise SystemExit("관리자 비밀번호는 대화형 로컬 터미널에서 설정하세요.")
    password = getpass.getpass("관리자 비밀번호 (14~128자): ")
    if not 14 <= len(password) <= 128:
        raise SystemExit("관리자 비밀번호는 14~128자여야 합니다.")
    if password != getpass.getpass("관리자 비밀번호 확인: "):
        raise SystemExit("비밀번호가 일치하지 않습니다.")
    salt = secrets.token_bytes(16)
    verifier = password_hash(password, salt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            json.dump({"salt": salt.hex(), "verifier": verifier.hex()}, temporary)
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    print("관리자 계정 admin을 설정했습니다.")


def read_credentials(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        salt = bytes.fromhex(data["salt"])
        verifier = bytes.fromhex(data["verifier"])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if len(salt) == 16 and len(verifier) == 32:
        return salt, verifier
    return None


class AdminAuth:
    """Require a local password verifier and keep session tokens in memory."""

    def __init__(self, path: Path):
        self.path = path
        self.credentials = read_credentials(path)
        self.sessions = {}
        self.failed_logins = deque()
        self.lock = threading.Lock()

    def refresh_credentials(self):
        credentials = read_credentials(self.path)
        if credentials != self.credentials:
            self.credentials = credentials
            self.sessions.clear()
            self.failed_logins.clear()

    def login(self, username, password, previous_token):
        with self.lock:
            self.refresh_credentials()
            if self.credentials is None:
                return 503, None
            now = time.monotonic()
            while self.failed_logins and self.failed_logins[0] <= now - LOGIN_WINDOW:
                self.failed_logins.popleft()
            if len(self.failed_logins) >= MAX_FAILED_LOGINS:
                return 429, None

            if isinstance(password, str) and 14 <= len(password) <= 128:
                salt, verifier = self.credentials
                try:
                    valid_password = secrets.compare_digest(password_hash(password, salt), verifier)
                except UnicodeEncodeError:
                    valid_password = False
                if valid_password and username == USERNAME:
                    self.failed_logins.clear()
                    self.sessions = {
                        token: expiry for token, expiry in self.sessions.items()
                        if expiry > now and token != previous_token
                    }
                    if len(self.sessions) >= MAX_SESSIONS:
                        del self.sessions[next(iter(self.sessions))]
                    token = secrets.token_urlsafe(32)
                    self.sessions[token] = now + SESSION_SECONDS
                    return 200, token

            self.failed_logins.append(now)
            return 401, None

    def is_authenticated(self, token):
        with self.lock:
            self.refresh_credentials()
            return self.sessions.get(token, 0) > time.monotonic()

    def logout(self, token):
        with self.lock:
            self.sessions.pop(token, None)
