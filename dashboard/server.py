"""Dependency-free team dashboard backend."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


# ---------------------------------------------------------------------
# Paths / Configuration
# ---------------------------------------------------------------------

ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()

STATIC = Path(__file__).parent / "static"

DATA = Path(
    os.getenv(
        "DASHBOARD_DATA",
        ROOT / ".dashboard-data",
    )
)

NOTES = DATA / "notes.json"

OUTPUT = Path(
    os.getenv("OUTPUT_DIR", ROOT / "output")
).resolve()

ARTIFACT_STORAGE = os.getenv("ARTIFACT_STORAGE_PATH", "").strip()


TASKS = {
    "train": {
        "label": "모델 학습",
        "description": "선택한 LightGCN 버전을 학습합니다.",
        "default": "v4_knowledge_bpr",
        "variants": {
            "v1": ("V1 LightGCN", "code/model/train.py"),
            "v1_split": ("V1 Split", "code/model/train_split.py"),
            "v2": ("V2 LightGCN", "code/model/train_v2.py"),
            "v2_split": ("V2 Split", "code/model/train_v2_split.py"),
            "v3_bpr": ("V3 BPR", "code/model/train_v3_bpr.py"),
            "v4_knowledge_bpr": (
                "V4 Knowledge BPR",
                "code/model/train_v4_knowledge_bpr.py",
            ),
        },
    },
    "benchmark": {
        "label": "통합 벤치마크",
        "description": "선택한 평가 방식으로 모델 성능을 비교합니다.",
        "default": "v2_zero_leakage",
        "variants": {
            "v1": ("V1 통합 벤치마크", "code/test/run_benchmark.py"),
            "v2_zero_leakage": (
                "V2 Zero-Leakage",
                "code/test/run_benchmark_v2.py",
            ),
        },
    },
    "inference": {
        "label": "자연어 추천",
        "description": "자연어 논문 추천 기능입니다.",
        "interactive": True,
    },
}


# ---------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------

jobs: dict[str, dict] = {}
job_processes: dict[str, subprocess.Popen] = {}
jobs_lock = threading.Lock()


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def now() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    timeout: int = 15,
) -> tuple[int, str, str]:
    """Run a command inside the project root."""

    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        return (
            process.returncode,
            process.stdout.strip(),
            process.stderr.strip(),
        )

    except Exception as error:
        return 1, "", str(error)


def gpu_diagnostics() -> tuple[bool, list[str]]:
    """Return CUDA availability and user-facing PyTorch diagnostics."""

    code, stdout, stderr = run(
        [
            "python",
            "-c",
            (
                "import sys, torch; "
                "available=torch.cuda.is_available(); "
                "print(f'Python: {sys.executable}'); "
                "print(f'PyTorch: {torch.__version__}'); "
                "print(f'PyTorch CUDA runtime: {torch.version.cuda or \"없음 (CPU 빌드)\"}'); "
                "print(f'torch.cuda.is_available(): {available}'); "
                "print(f'CUDA device count: {torch.cuda.device_count()}'); "
                "print(f'CUDA device: {torch.cuda.get_device_name(0) if available else \"없음\"}')"
            ),
        ],
        timeout=30,
    )

    lines = [line for line in stdout.splitlines() if line]
    if stderr:
        lines.extend(line for line in stderr.splitlines() if line)
    if code != 0:
        lines.append(f"CUDA 진단 명령 종료 코드: {code}")

    available = code == 0 and any(
        line == "torch.cuda.is_available(): True" for line in lines
    )
    return available, lines


def gpu_available() -> bool:
    """Return whether PyTorch can see a CUDA GPU."""

    return gpu_diagnostics()[0]


def discover_scripts() -> list[dict]:
    """Discover runnable Python files from the mounted working tree."""

    scripts = []

    for path in sorted((ROOT / "code").rglob("*.py")):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue

        scripts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "name": path.stem,
                "group": path.parent.relative_to(ROOT / "code").as_posix(),
            }
        )

    return scripts


def resolve_script(relative_path: str) -> Path | None:
    """Resolve a discovered script without allowing path traversal."""

    target = (ROOT / relative_path).resolve()
    code_root = (ROOT / "code").resolve()

    if (
        target.suffix != ".py"
        or not target.is_file()
        or code_root not in target.parents
    ):
        return None

    return target


def storage_info() -> dict:
    """Describe the optional shared artifact directory."""

    external = Path(ARTIFACT_STORAGE).resolve() if ARTIFACT_STORAGE else None

    return {
        "configured": external is not None,
        "output": str(OUTPUT),
        "external": str(external) if external else "",
        "external_exists": bool(external and external.exists()),
    }


def copy_artifacts(source: Path, destination: Path) -> int:
    """Copy artifact files while preserving their relative paths."""

    if not source.exists():
        raise FileNotFoundError(str(source))

    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    for path in source.rglob("*"):
        if not path.is_file():
            continue

        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1

    return copied


# ---------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------

def git_info() -> dict:
    """Collect repository, branch, commit, and working-tree information."""

    _, branch, _ = run(
        [
            "git",
            "branch",
            "--show-current",
        ]
    )

    _, dirty, _ = run(
        [
            "git",
            "status",
            "--porcelain",
        ]
    )

    _, remote, _ = run(
        [
            "git",
            "remote",
            "get-url",
            "origin",
        ]
    )

    _, raw_commits, git_error = run(
        [
            "git",
            "log",
            "-10",
            "--pretty=format:%h%x1f%an%x1f%aI%x1f%s%x1e",
        ]
    )

    commits = parse_git_commits(raw_commits)

    _, raw_branches, _ = run(
        [
            "git",
            "for-each-ref",
            "--sort=-committerdate",
            "--format=%(refname:short)|%(objectname:short)|%(subject)",
            "refs/heads",
            "refs/remotes",
        ]
    )

    branches = parse_git_branches(raw_branches)

    if not branch:
        try:
            branch, remote, commits, branches = load_git_fallback(
                remote=remote,
                commits=commits,
                branches=branches,
            )

        except OSError as fallback_error:
            git_error = str(fallback_error)

    return {
        "branch": branch or "unknown",
        "dirty": bool(dirty),
        "changes": len(dirty.splitlines()),
        "remote": remote,
        "commits": commits,
        "branches": branches,
        "error": git_error,
    }


def branch_commits(branch: str) -> tuple[int, list[dict], str]:
    """Return recent commits for an existing local or remote branch."""

    valid_names = {item["name"] for item in git_info()["branches"]}

    if branch not in valid_names:
        return 1, [], "존재하지 않는 브랜치입니다."

    code, output, error = run(
        [
            "git",
            "log",
            branch,
            "-20",
            "--pretty=format:%h%x1f%an%x1f%aI%x1f%s%x1e",
        ]
    )

    return code, parse_git_commits(output), error


def switch_branch(branch: str) -> tuple[int, str]:
    """Switch to a known branch, creating a local tracking branch if needed."""

    info = git_info()
    valid_names = {item["name"] for item in info["branches"]}

    if branch not in valid_names:
        return 1, "존재하지 않는 브랜치입니다."

    if info["dirty"]:
        return 1, "미커밋 변경이 있어 브랜치를 전환할 수 없습니다."

    if branch.startswith("origin/"):
        local_name = branch.split("/", 1)[1]
        local_names = {
            name for name in valid_names if not name.startswith("origin/")
        }
        command = (
            ["git", "switch", local_name]
            if local_name in local_names
            else ["git", "switch", "--track", branch]
        )
    else:
        command = ["git", "switch", branch]

    code, stdout, stderr = run(command, timeout=90)
    return code, stdout or stderr


def parse_git_commits(raw: str) -> list[dict]:
    """Parse formatted output from git log."""

    commits = []

    if not raw:
        return commits

    rows = raw.strip("\x1e\n").split("\x1e")

    for row in rows:
        parts = row.strip().split("\x1f")

        if len(parts) != 4:
            continue

        commits.append(
            dict(
                zip(
                    (
                        "hash",
                        "author",
                        "date",
                        "message",
                    ),
                    parts,
                )
            )
        )

    return commits


def parse_git_branches(raw: str) -> list[dict]:
    """Parse local and remote branch information."""

    branches = []

    for row in raw.splitlines():
        parts = row.split("|", 2)

        if len(parts) != 3:
            continue

        if parts[0].endswith("/HEAD"):
            continue

        branches.append(
            dict(
                zip(
                    (
                        "name",
                        "hash",
                        "message",
                    ),
                    parts,
                )
            )
        )

    return branches


def load_git_fallback(
    remote: str,
    commits: list[dict],
    branches: list[dict],
) -> tuple[str, str, list[dict], list[dict]]:
    """
    Read Git metadata directly from .git if Git commands fail.

    This is mainly useful in limited environments where the repository
    metadata exists but normal Git commands cannot determine HEAD.
    """

    git_dir = ROOT / ".git"

    head = (git_dir / "HEAD").read_text(
        encoding="utf-8",
    ).strip()

    if head.startswith("ref: "):
        branch = head.rsplit("/", 1)[-1]
    else:
        branch = head[:8]

    if not remote:
        remote = read_git_remote_from_config(git_dir)

    if not branches:
        branches = read_git_refs(git_dir)

    if not commits:
        commits = read_git_head_log(git_dir)

    return branch, remote, commits, branches


def read_git_remote_from_config(git_dir: Path) -> str:
    """Read the first remote URL from .git/config."""

    config_path = git_dir / "config"

    config = config_path.read_text(
        encoding="utf-8",
    )

    urls = re.findall(
        r"^\s*url\s*=\s*(.+)$",
        config,
        re.MULTILINE,
    )

    if not urls:
        return ""

    return urls[0].strip()


def read_git_refs(git_dir: Path) -> list[dict]:
    """Read branch refs directly from the .git directory."""

    branches = []

    ref_roots = (
        git_dir / "refs" / "heads",
        git_dir / "refs" / "remotes",
    )

    for base in ref_roots:
        if not base.exists():
            continue

        for ref in base.rglob("*"):
            if not ref.is_file():
                continue

            if ref.name == "HEAD":
                continue

            branches.append(
                {
                    "name": ref.relative_to(base).as_posix(),
                    "hash": ref.read_text().strip()[:8],
                    "message": "",
                }
            )

    return branches


def read_git_head_log(git_dir: Path) -> list[dict]:
    """Read recent commits from .git/logs/HEAD."""

    log_path = git_dir / "logs" / "HEAD"

    if not log_path.exists():
        return []

    rows = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()[-20:]

    commits = []

    for row in reversed(rows):
        meta, _, message = row.partition("\t")

        match = re.match(
            r"\S+\s+(\S+)\s+(.+?)\s+<[^>]+>\s+(\d+)\s+[+-]\d+",
            meta,
        )

        if not match:
            continue

        commit_hash = match.group(1)[:8]
        author = match.group(2)
        timestamp = int(match.group(3))

        commits.append(
            {
                "hash": commit_hash,
                "author": author,
                "date": datetime.fromtimestamp(
                    timestamp,
                    timezone.utc,
                ).isoformat(),
                "message": message or "repository update",
            }
        )

    return commits


# ---------------------------------------------------------------------
# GitHub Issues
# ---------------------------------------------------------------------

def issue_data() -> dict:
    """Load open GitHub issues for the current repository."""

    remote = git_info()["remote"]
    repository = os.getenv("GITHUB_REPOSITORY")

    if not repository:
        repository = repository_from_remote(remote)

    if not repository:
        return {
            "repo": None,
            "issues": [],
            "error": "GitHub 저장소를 찾지 못했습니다.",
        }

    request = urllib.request.Request(
        (
            f"https://api.github.com/repos/{repository}/issues"
            "?state=open&per_page=50"
        ),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "team-dashboard",
        },
    )

    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        request.add_header(
            "Authorization",
            f"Bearer {github_token}",
        )

    try:
        with urllib.request.urlopen(
            request,
            timeout=8,
        ) as response:
            raw_issues = json.load(response)

    except Exception as error:
        return {
            "repo": repository,
            "issues": [],
            "error": str(error),
        }

    issues = []

    for issue in raw_issues:
        if "pull_request" in issue:
            continue

        labels = [
            label["name"]
            for label in issue.get("labels", [])
        ]

        issues.append(
            {
                "number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "labels": labels,
                "severity": issue_severity(
                    issue.get("labels", [])
                ),
            }
        )

    severity_order = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }

    issues.sort(
        key=lambda issue: severity_order[
            issue["severity"]
        ]
    )

    return {
        "repo": repository,
        "issues": issues,
        "error": None,
    }


def repository_from_remote(remote: str) -> str | None:
    """Extract owner/repository from a GitHub remote URL."""

    if not remote:
        return None

    match = re.search(
        r"github\.com[/:]([^/]+/[^/.]+)(?:\.git)?$",
        remote,
    )

    if not match:
        return None

    return match.group(1)


def issue_severity(labels: list[dict]) -> str:
    """Infer issue severity from GitHub labels."""

    label_text = " ".join(
        label["name"].lower()
        for label in labels
    )

    severity_labels = (
        (
            "critical",
            (
                "critical",
                "blocker",
                "p0",
                "긴급",
            ),
        ),
        (
            "high",
            (
                "high",
                "p1",
                "높음",
            ),
        ),
        (
            "low",
            (
                "low",
                "p3",
                "낮음",
            ),
        ),
    )

    for severity, keywords in severity_labels:
        if any(
            keyword in label_text
            for keyword in keywords
        ):
            return severity

    return "medium"


# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------

def read_notes() -> list[dict]:
    """Read dashboard notes from disk."""

    if not NOTES.exists():
        return []

    try:
        return json.loads(
            NOTES.read_text(
                encoding="utf-8",
            )
        )

    except (OSError, json.JSONDecodeError):
        return []


def save_notes(items: list[dict]) -> None:
    """Write dashboard notes atomically."""

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = NOTES.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            items,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(NOTES)


# ---------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------

def docs() -> list[dict]:
    """Return Markdown documents available in the project."""

    paths = [
        ROOT / "readme.md",
        *sorted(
            (ROOT / "docs").glob("*.md")
        ),
    ]

    return [
        {
            "name": path.name,
            "path": path.relative_to(ROOT).as_posix(),
        }
        for path in paths
        if path.exists()
    ]


def tree() -> list[dict]:
    """Return selected project directories and file counts."""

    sections = (
        (
            "code/data_collection",
            "데이터 수집",
        ),
        (
            "code/data_analysis",
            "LLM 지식 추출",
        ),
        (
            "code/model",
            "모델 학습",
        ),
        (
            "code/test",
            "평가와 추론",
        ),
        (
            "subdataset",
            "전처리 데이터",
        ),
        (
            "output",
            "가중치와 결과",
        ),
        (
            "docs",
            "연구 문서",
        ),
    )

    result = []

    for name, description in sections:
        path = ROOT / name

        file_count = 0

        if path.exists():
            file_count = sum(
                item.is_file()
                for item in path.rglob("*")
            )

        result.append(
            {
                "path": name,
                "description": description,
                "files": file_count,
            }
        )

    return result


# ---------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------

def worker(
    job_id: str,
    command: list[str],
) -> None:
    """Run a dashboard task and stream its output into job state."""

    with jobs_lock:
        if jobs[job_id].get("cancel_requested"):
            jobs[job_id].update(status="cancelled", finished_at=now())
            return
        jobs[job_id]["status"] = "running"

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        with jobs_lock:
            job_processes[job_id] = process
            cancel_requested = jobs[job_id].get("cancel_requested", False)

        if cancel_requested:
            process.terminate()

        for line in process.stdout or []:
            with jobs_lock:
                jobs[job_id]["log"].append(
                    line.rstrip()
                )

                jobs[job_id]["log"] = (
                    jobs[job_id]["log"][-2000:]
                )

        exit_code = process.wait()

        with jobs_lock:
            cancelled = jobs[job_id].get("cancel_requested", False)
            jobs[job_id].update(
                status=(
                    "cancelled"
                    if cancelled
                    else "success"
                    if exit_code == 0
                    else "failed"
                ),
                exit_code=exit_code,
                finished_at=now(),
            )
            job_processes.pop(job_id, None)

    except Exception as error:
        with jobs_lock:
            jobs[job_id].update(
                status="failed",
                error=str(error),
                finished_at=now(),
            )
            jobs[job_id]["log"].append(f"[실행 오류] {error}")
            job_processes.pop(job_id, None)


# ---------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    """Serve dashboard static assets and JSON API endpoints."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            directory=str(STATIC),
            **kwargs,
        )

    def send_json(
        self,
        data,
        status: int = 200,
    ) -> None:
        """Serialize and send a JSON response."""

        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.end_headers()
        self.wfile.write(body)

    def payload(self) -> dict:
        """Read and decode a JSON request body."""

        content_length = int(
            self.headers.get(
                "Content-Length",
                0,
            )
        )

        raw_body = self.rfile.read(
            content_length
        )

        return json.loads(
            raw_body or b"{}"
        )

    def do_GET(self) -> None:
        """Handle GET requests."""

        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/api/health":
            return self.send_json(
                {
                    "ok": True,
                    "time": now(),
                }
            )

        if path == "/api/git":
            return self.send_json(
                git_info()
            )

        if path == "/api/git/commits":
            branch = parse_qs(parsed_url.query).get("branch", [""])[0]
            code, commits, error = branch_commits(branch)
            return self.send_json(
                {"branch": branch, "commits": commits, "error": error},
                200 if code == 0 else 404,
            )

        if path == "/api/issues":
            return self.send_json(
                issue_data()
            )

        if path == "/api/notes":
            return self.send_json(
                read_notes()
            )

        if path == "/api/docs":
            return self.send_json(
                {
                    "documents": docs(),
                    "tree": tree(),
                }
            )

        if path == "/api/doc":
            return self.handle_document_request(
                parsed_url
            )

        if path == "/api/tasks":
            return self.handle_tasks_request()

        if path == "/api/scripts":
            return self.send_json({"scripts": discover_scripts()})

        if path == "/api/storage":
            return self.send_json(storage_info())

        if path.startswith("/api/jobs/"):
            return self.handle_job_request(
                path
            )

        return super().do_GET()

    def do_POST(self) -> None:
        """Handle POST requests."""

        try:
            data = self.payload()

        except (
            json.JSONDecodeError,
            ValueError,
        ):
            return self.send_json(
                {
                    "error": "잘못된 요청입니다.",
                },
                400,
            )

        if self.path == "/api/notes":
            return self.handle_create_note(
                data
            )

        if self.path == "/api/git/pull":
            return self.handle_git_pull(data)

        if self.path == "/api/git/switch":
            return self.handle_git_switch(data)

        if self.path == "/api/scripts/run":
            return self.handle_run_script(data)

        if self.path.startswith("/api/jobs/") and self.path.endswith("/stop"):
            return self.handle_stop_job(self.path)

        if self.path == "/api/storage/sync":
            return self.handle_storage_sync(data)

        if self.path.startswith(
            "/api/tasks/"
        ):
            return self.handle_start_task(data)

        return self.send_json(
            {
                "error": "지원하지 않는 요청입니다.",
            },
            404,
        )

    def handle_document_request(
        self,
        parsed_url,
    ) -> None:
        """Serve one Markdown document."""

        relative_path = parse_qs(
            parsed_url.query
        ).get(
            "path",
            [""],
        )[0]

        target = (
            ROOT / relative_path
        ).resolve()

        readme_path = (
            ROOT / "readme.md"
        ).resolve()

        docs_path = (
            ROOT / "docs"
        ).resolve()

        allowed = (
            target == readme_path
            or docs_path in target.parents
        )

        if (
            allowed
            and target.suffix.lower() == ".md"
            and target.exists()
        ):
            return self.send_json(
                {
                    "path": relative_path,
                    "content": target.read_text(
                        encoding="utf-8"
                    ),
                }
            )

        return self.send_json(
            {
                "error": "문서를 찾지 못했습니다.",
            },
            404,
        )

    def handle_tasks_request(self) -> None:
        """Return task metadata and recent jobs."""

        task_data = {}

        for key, task in TASKS.items():
            public_task = {
                name: value
                for name, value in task.items()
                if name not in {"command", "variants"}
            }
            public_task["variants"] = [
                {"id": variant_id, "label": value[0], "path": value[1]}
                for variant_id, value in task.get("variants", {}).items()
            ]
            task_data[key] = public_task

        with jobs_lock:
            recent_jobs = list(
                jobs.values()
            )[-10:]

        return self.send_json(
            {
                "tasks": task_data,
                "jobs": recent_jobs,
            }
        )

    def handle_job_request(
        self,
        path: str,
    ) -> None:
        """Return one background job."""

        job_id = path.rsplit(
            "/",
            1,
        )[-1]

        with jobs_lock:
            job = jobs.get(job_id)

        if job:
            return self.send_json(job)

        return self.send_json(
            {
                "error": "작업이 없습니다.",
            },
            404,
        )

    def handle_stop_job(self, path: str) -> None:
        """Request termination of one queued or running background job."""

        job_id = path.removesuffix("/stop").rsplit("/", 1)[-1]

        with jobs_lock:
            job = jobs.get(job_id)

            if not job:
                return self.send_json({"error": "작업이 없습니다."}, 404)

            if job["status"] in {"success", "failed", "cancelled"}:
                return self.send_json(job)

            job["cancel_requested"] = True
            job["status"] = "stopping"
            job["log"].append("[중지 요청] 사용자가 작업 중지를 요청했습니다.")
            process = job_processes.get(job_id)

        if process and process.poll() is None:
            process.terminate()

        return self.send_json(job, 202)

    def handle_create_note(
        self,
        data: dict,
    ) -> None:
        """Create a dashboard note."""

        author = str(
            data.get(
                "author",
                "",
            )
        ).strip()[:40]

        content = str(
            data.get(
                "content",
                "",
            )
        ).strip()[:2000]

        if not author or not content:
            return self.send_json(
                {
                    "error": "작성자와 내용을 입력하세요.",
                },
                400,
            )

        item = {
            "id": uuid.uuid4().hex,
            "author": author,
            "content": content,
            "created_at": now(),
        }

        items = read_notes()
        items.insert(0, item)

        save_notes(
            items[:200]
        )

        return self.send_json(
            item,
            201,
        )

    def handle_git_pull(self, data: dict) -> None:
        """Pull the selected branch with fast-forward-only semantics."""

        branch = str(data.get("branch", "")).strip()

        fetch_code, _, fetch_error = run(
            ["git", "fetch", "origin", "--prune"],
            timeout=90,
        )
        if fetch_code != 0:
            return self.send_json({"error": fetch_error}, 409)

        if branch and branch != git_info()["branch"]:
            code, message = switch_branch(branch)
            if code != 0:
                return self.send_json({"error": message}, 409)

        code, stdout, stderr = run(
            [
                "git",
                "pull",
                "--ff-only",
            ],
            timeout=90,
        )

        response = {
            "ok": code == 0,
            "output": stdout or stderr,
        }

        return self.send_json(
            response,
            200 if code == 0 else 409,
        )

    def handle_git_switch(self, data: dict) -> None:
        """Switch the dashboard working tree to a selected branch."""

        branch = str(data.get("branch", "")).strip()
        code, output = switch_branch(branch)
        return self.send_json(
            {"ok": code == 0, "output": output, "branch": git_info()["branch"]},
            200 if code == 0 else 409,
        )

    def start_job(self, task_key: str, label: str, command: list[str]) -> None:
        """Create and start one background process."""

        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "task": task_key,
            "label": label,
            "command": command,
            "status": "queued",
            "created_at": now(),
            "log": [],
        }

        with jobs_lock:
            jobs[job_id] = job

        threading.Thread(
            target=worker,
            args=(job_id, command),
            daemon=True,
        ).start()

        return self.send_json(job, 202)

    def handle_run_script(self, data: dict) -> None:
        """Run a Python script discovered from the mounted code directory."""

        relative_path = str(data.get("path", "")).strip()
        script = resolve_script(relative_path)

        if script is None:
            return self.send_json({"error": "실행 가능한 스크립트가 아닙니다."}, 400)

        return self.start_job(
            "script",
            f"스크립트: {relative_path}",
            ["python", "-u", relative_path],
        )

    def handle_storage_sync(self, data: dict) -> None:
        """Synchronize output with a configured filesystem-backed store."""

        if not ARTIFACT_STORAGE:
            return self.send_json(
                {"error": "ARTIFACT_STORAGE_PATH가 설정되지 않았습니다."},
                409,
            )

        direction = str(data.get("direction", "pull"))
        external = Path(ARTIFACT_STORAGE).resolve()

        try:
            if direction == "pull":
                copied = copy_artifacts(external, OUTPUT)
            elif direction == "push":
                copied = copy_artifacts(OUTPUT, external)
            else:
                return self.send_json({"error": "잘못된 동기화 방향입니다."}, 400)
        except OSError as error:
            return self.send_json({"error": str(error)}, 409)

        return self.send_json(
            {"ok": True, "direction": direction, "files": copied}
        )

    def handle_start_task(self, data: dict) -> None:
        """Start one configured background task."""

        task_key = self.path.rsplit(
            "/",
            1,
        )[-1]

        task = TASKS.get(task_key)

        if not task:
            return self.send_json(
                {
                    "error": "등록되지 않은 작업입니다.",
                },
                404,
            )

        if task.get("interactive"):
            return self.send_json(
                {
                    "error": (
                        "대화형 추천 UI는 "
                        "실사용 탭에서 구현 예정입니다."
                    )
                },
                409,
            )

        variants = task.get("variants", {})
        variant_id = str(data.get("variant") or task.get("default", ""))
        variant = variants.get(variant_id)

        if not variant:
            return self.send_json({"error": "지원하지 않는 실행 버전입니다."}, 400)

        require_gpu = (
            os.getenv(
                "REQUIRE_GPU",
                "true",
            ).lower()
            in {
                "1",
                "true",
                "yes",
            }
        )

        gpu_ok, gpu_log = gpu_diagnostics()

        if require_gpu and not gpu_ok:
            return self.send_json(
                {
                    "error": (
                        "CUDA GPU를 찾지 못했습니다. "
                        "Colab에서 실행 버튼을 사용하세요."
                    ),
                    "log": [
                        "$ CUDA 사용 가능 여부를 확인했습니다.",
                        *gpu_log,
                    ],
                },
                409,
            )

        return self.start_job(
            task_key,
            variant[0],
            ["python", "-u", variant[1]],
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    """Start the dashboard server."""

    DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    port = int(
        os.getenv(
            "DASHBOARD_PORT",
            "8080",
        )
    )

    print(
        f"Dashboard http://0.0.0.0:{port}"
    )

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            port,
        ),
        Handler,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
