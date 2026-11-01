"""GPU memory polling (NVML -> nvidia-smi -> disabled).

Server engines run in another process, so device-level used memory sampled
over time is the only engine-agnostic measure. Each request row gets the max
sample observed during its lifetime; each run gets its overall peak.
Note: vLLM-Omni pre-allocates KV cache by ``gpu_memory_utilization``, so its
"used" memory reflects the reservation, not live demand; report both the
reservation and torch peak allocation (official engine) where available.
"""

from __future__ import annotations

import bisect
import shutil
import subprocess
import threading
import time


class GpuMemoryPoller:
    def __init__(self, device_index: int = 0, interval_s: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self.backend = "none"
        self._times: list[float] = []
        self._used_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_handle = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._pynvml = pynvml
            self.backend = "nvml"
            return
        except Exception:  # noqa: BLE001
            pass
        if shutil.which("nvidia-smi"):
            if self._smi_sample() is not None:
                self.backend = "nvidia-smi"

    def _smi_sample(self) -> float | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.device_index)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def sample(self) -> float | None:
        if self.backend == "nvml":
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return info.used / 2**20
        if self.backend == "nvidia-smi":
            return self._smi_sample()
        return None

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self.sample()
            if v is not None:
                self._times.append(time.perf_counter())
                self._used_mb.append(v)
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuMemoryPoller":
        if self.available and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def peak_mb(self) -> float | None:
        return max(self._used_mb) if self._used_mb else None

    def max_between(self, t0: float, t1: float) -> float | None:
        """Max sample in [t0, t1] (perf_counter); nearest earlier sample if none fall inside."""
        if not self._times:
            return None
        i = bisect.bisect_left(self._times, t0)
        j = bisect.bisect_right(self._times, t1)
        if j > i:
            return max(self._used_mb[i:j])
        return self._used_mb[max(0, i - 1)]

    def __enter__(self) -> "GpuMemoryPoller":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
