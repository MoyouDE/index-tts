"""Best-effort WDDM process GPU memory sampling (not a peak certificate)."""

import ctypes
import os
import threading


class ProcessGpuMemorySampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if os.name != "nt":
            self.error = "WDDM counters are only available on Windows"
            return self
        try:
            self._open()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.error = str(exc)
            if getattr(self, "query", None):
                self.pdh.PdhCloseQuery(self.query)
                self.query = None
        return self

    def _open(self):
        from ctypes import wintypes
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_size_t,
                                                  ctypes.POINTER(ctypes.c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                                         ctypes.POINTER(wintypes.DWORD),
                                                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
            raise RuntimeError("Cannot open PDH query")
        self.counters = []
        for name in ("Dedicated Usage", "Shared Usage"):
            counter = ctypes.c_void_p()
            status = self.pdh.PdhAddEnglishCounterW(
                self.query, f"\\GPU Process Memory(*)\\{name}", 0, ctypes.byref(counter))
            if status:
                raise RuntimeError(f"Cannot read WDDM {name}: {status & 0xffffffff:#x}")
            self.counters.append(counter)

    def _value(self, counter):
        from ctypes import wintypes
        class Value(ctypes.Structure):
            _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]
        class Item(ctypes.Structure):
            _fields_ = [("name", wintypes.LPWSTR), ("value", Value)]
        size, count = wintypes.DWORD(), wintypes.DWORD()
        self.pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), buffer):
            return None
        items = ctypes.cast(buffer, ctypes.POINTER(Item))
        matching = [items[index].value.value for index in range(count.value)
                    if f"pid_{os.getpid()}_" in items[index].name
                    and items[index].value.status in (0, 1)]
        return int(sum(matching)) if matching else None

    def _sample(self):
        try:
            while not self._stop.is_set():
                if not self.pdh.PdhCollectQueryData(self.query):
                    self.samples.append(tuple(self._value(counter) for counter in self.counters))
                self._stop.wait(self.interval)
        except Exception as exc:
            self.error = str(exc)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
            self.pdh.PdhCloseQuery(self.query)
        return self.report()

    def report(self):
        def peak(index):
            values = [sample[index] for sample in self.samples if sample[index] is not None]
            return max(values) if values else None
        return {"method": "WDDM PDH process counters", "intervalMs": self.interval * 1000,
                "samples": len(self.samples), "dedicatedSampledPeakBytes": peak(0),
                "sharedSampledPeakBytes": peak(1), "strictPeakCertified": False, "error": self.error}
