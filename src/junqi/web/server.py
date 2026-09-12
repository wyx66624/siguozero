"""Loopback-only console. Training is monitored; game workers cannot use GPUs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from .monitor import Monitor, clean_numbers
from .service_control import write_service_metadata


class Worker:
    def __init__(self, root, threads, log_directory):
        self.id = secrets.token_hex(16)
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.closed = False
        self.busy = False
        self.last_state = None
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none",
                           HIP_VISIBLE_DEVICES="", ASCEND_RT_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                           OPENBLAS_NUM_THREADS=str(threads), PYTHONPATH=str(root / "src"),
                           PYTHONUNBUFFERED="1")
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(key, None)
        log_directory.mkdir(parents=True, exist_ok=True)
        with (log_directory / f"cpu-{self.id}.log").open("a") as log:
            self.process = subprocess.Popen([sys.executable, "-m", "junqi.web.cpu_game"],
                cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True, encoding="utf-8", bufsize=1, start_new_session=True)
        self.reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-reply")

    def request(self, payload):
        if not self.lock.acquire(timeout=1):
            raise ValueError("模型正在思考，请等待当前操作完成")
        self.busy = True
        try:
            if self.closed or self.process.poll() is not None:
                raise ValueError("CPU 对弈进程已退出，请重新开局")
            self.last_used = time.monotonic()
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            future = self.reader.submit(self.process.stdout.readline)
            try:
                line = future.result(timeout=180)
            except FutureTimeout:
                self.close()
                raise ValueError("CPU 推理超时，已释放本局进程，请重新开局")
            if not line:
                raise ValueError("CPU 对弈进程意外退出，请检查服务日志")
            result = json.loads(line)
            if not result.get("ok"):
                raise ValueError(result.get("error", "CPU 推理失败"))
            self.last_state = result["state"]
            self.last_used = time.monotonic()
            return {"session_id": self.id, **self.last_state}
        finally:
            self.busy = False
            self.lock.release()

    def close(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.reader.shutdown(wait=False, cancel_futures=True)
        self.process.stdin.close()
        self.process.stdout.close()


class Application:
    def __init__(self, config, root):
        self.root = Path(root).resolve()
        self.monitor = Monitor(config, self.root)
        self.threads = int(config.get("cpu_threads", 2))
        self.max_games = int(config.get("max_games", 2))
        if not 1 <= self.threads <= 4 or not 1 <= self.max_games <= 4:
            raise ValueError("CPU threads and concurrent games must be in 1..4")
        self.workers = {}
        self.lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.cached = None
        self.cached_at = 0
        self.stop = threading.Event()
        self.reaper = threading.Thread(target=self._reap, daemon=True)
        self.reaper.start()

    def _reap(self):
        while not self.stop.wait(30):
            with self.lock:
                stale = [key for key, worker in self.workers.items() if not worker.busy and
                         (worker.closed or worker.process.poll() is not None or time.monotonic() - worker.last_used > 1800)]
                for key in stale:
                    self.workers.pop(key).close()

    def status(self):
        with self.cache_lock:
            if time.monotonic() - self.cached_at > 5 or self.cached is None:
                self.cached = self.monitor.status()
                self.cached_at = time.monotonic()
            result = dict(self.cached)
        with self.lock:
            result["cpu_play"] = {"device": "cpu", "threads_per_game": self.threads,
                "max_games": self.max_games, "sessions": [{"pid": w.process.pid, "busy": w.busy,
                    "alive": w.process.poll() is None,
                    "model": (w.last_state or {}).get("model")} for w in self.workers.values()]}
        return result

    def create_game(self, data):
        entry = self.monitor.resolve_checkpoint(data.get("checkpoint_id"))
        seat = data.get("seat", 0)
        count = 2 if entry["mode"] == "two_player" else 4
        if type(seat) is not int or not 0 <= seat < count:
            raise ValueError("无效座位")
        temperature = data.get("temperature", 0.7)
        if type(temperature) not in (int, float) or not 0.1 <= temperature <= 2:
            raise ValueError("温度须在 0.1 至 2 之间")
        seed = data.get("seed", secrets.randbelow(2**31))
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("无效随机种子")
        with self.lock:
            if len(self.workers) >= self.max_games:
                raise ValueError("已达到 CPU 棋局上限，请先结束旧棋局")
            worker = Worker(self.root, self.threads, self.root / "output/local_console")
            self.workers[worker.id] = worker
        try:
            return worker.request({"op": "new", "checkpoint": str(entry["_path"]),
                "mode": entry["mode"], "dead_rules_enabled": entry["dead_rules_enabled"],
                "seat": seat, "temperature": temperature, "seed": seed, "threads": self.threads})
        except Exception:
            with self.lock:
                self.workers.pop(worker.id, None)
            worker.close()
            raise

    def game_command(self, data, op):
        with self.lock:
            worker = self.workers.get(data.get("session_id"))
        if worker is None:
            raise ValueError("棋局不存在或空闲超过 30 分钟，请重新开局")
        if op == "close":
            if worker.busy:
                raise ValueError("模型正在思考，请稍后结束棋局")
            with self.lock:
                self.workers.pop(worker.id, None)
            worker.close()
            return {"closed": True}
        return worker.request({**data, "op": op})

    def close(self):
        self.stop.set()
        with self.lock:
            for worker in self.workers.values():
                worker.close()
            self.workers.clear()


class Handler(BaseHTTPRequestHandler):
    server_version = "SiguoZeroLocal/1"

    def _respond(self, status, content, content_type="application/json; charset=utf-8"):
        payload = json.dumps(clean_numbers(content), ensure_ascii=False, allow_nan=False).encode() if not isinstance(content, bytes) else content
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _local_request(self):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + host for host in allowed}:
            return False
        return self.headers.get("Sec-Fetch-Site") not in ("cross-site",)

    def do_GET(self):
        if not self._local_request():
            self._respond(403, {"error": "仅允许本地同源访问"})
            return
        path = urlsplit(self.path).path
        app = self.server.app
        if path == "/api/status":
            self._respond(200, app.status())
        elif path == "/api/models":
            self._respond(200, {"models": app.monitor.public_catalog()})
        elif path == "/api/health":
            self._respond(200, {"ok": True, "pid": os.getpid(), "gpu_inference": False,
                               "managed_by": os.environ.get("SIGUOZERO_MONITOR_SERVICE")})
        else:
            allowed = {"/": "index.html", "/play": "play.html", "/style.css": "style.css",
                       "/dashboard.js": "dashboard.js", "/play.js": "play.js"}
            name = allowed.get(path)
            if not name:
                self._respond(404, {"error": "页面不存在"})
                return
            file = Path(__file__).parent / "static" / name
            self._respond(200, file.read_bytes(), (mimetypes.guess_type(name)[0] or "text/plain") + "; charset=utf-8")

    def do_POST(self):
        if not self._local_request() or not self.headers.get("Content-Type", "").startswith("application/json"):
            self._respond(403, {"error": "仅允许本地同源 JSON 请求"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192:
                raise ValueError("无效请求长度")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("无效请求")
            path = urlsplit(self.path).path
            if path == "/api/game/new":
                result = self.server.app.create_game(data)
            elif path in {"/api/game/" + op for op in ("state", "move", "advance", "close", "replay")}:
                result = self.server.app.game_command(data, path.rsplit("/", 1)[1])
            else:
                self._respond(404, {"error": "接口不存在"})
                return
            self._respond(200, result)
        except (ValueError, OSError) as error:
            self._respond(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local_console.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-service-metadata", action="store_true", help="avoid changing launcher metadata for an isolated test server")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    app = Application(config, args.root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.app = app
    def stop_server(_signum, _frame):
        # shutdown() must run outside serve_forever's thread.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    if not args.no_service_metadata:
        write_service_metadata(args.root, args.config, server.server_port,
                               managed_by=os.environ.get("SIGUOZERO_MONITOR_SERVICE"))
    print(f"Local training console: http://localhost:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
