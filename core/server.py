"""Launch and manage the SGLang server as a subprocess."""

import subprocess
import sys
import time
import httpx


def launch_server(
    model_path: str = "/workspace/models/qed-nano",
    port: int = 30000,
    tp: int = 1,
    mem_fraction: float = 0.85,
    timeout: int = 300,
) -> subprocess.Popen:
    """Start the SGLang server and block until it's ready.

    Returns the Popen handle so the caller can terminate it later.
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", str(tp),
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction),
    ]
    print(f"[server] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    wait_for_ready(port=port, timeout=timeout)
    return proc


def wait_for_ready(port: int = 30000, timeout: int = 300) -> None:
    """Poll the SGLang health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                print(f"[server] Ready on port {port}")
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang server not ready after {timeout}s")


def shutdown_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate the server process."""
    if proc.poll() is None:
        print("[server] Shutting down...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[server] Stopped.")
