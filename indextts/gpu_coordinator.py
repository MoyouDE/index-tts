"""One serial GPU lane; model owners provide public release callbacks."""
import threading
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GpuCoordinator:
    def __init__(self):
        self._lock = threading.RLock()
        self._release = {}

    def bind(self, role, release):
        if role not in {"producer", "audition"}:
            raise ValueError("Unknown GPU role")
        with self._lock:
            self._release[role] = release

    @contextmanager
    def use(self, role, cancelled=None):
        if role not in {"producer", "audition"}:
            raise ValueError("Unknown GPU role")
        with self._lock:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("试听已取消" if role == "audition" else "制包已取消")
            logger.info("GPU operation acquired: %s", role)
            for other, release in self._release.items():
                if other != role:
                    logger.info("GPU owner release requested: %s -> %s", other, role)
                    release()
            try:
                yield
            finally:
                logger.info("GPU operation finished: %s", role)

    def unload(self, role):
        with self._lock:
            if role in self._release:
                self._release[role]()

    def close(self):
        with self._lock:
            for release in self._release.values():
                release()
