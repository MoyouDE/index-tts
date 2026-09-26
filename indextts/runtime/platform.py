"""Optional Windows calling-thread policy; never changes process affinity."""

import contextlib
import os


@contextlib.contextmanager
def performance_cores(enabled=False):
    if not enabled or os.name != "nt":
        yield
        return
    import ctypes
    import struct
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetSystemCpuSetInformation.argtypes = [ctypes.c_void_p, wintypes.ULONG,
                                                 ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG]
    kernel.GetThreadSelectedCpuSets.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG),
                                               wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
    kernel.GetProcessDefaultCpuSets.argtypes = kernel.GetThreadSelectedCpuSets.argtypes
    kernel.SetThreadSelectedCpuSets.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG]
    thread, process = kernel.GetCurrentThread(), kernel.GetCurrentProcess()

    def selected(function, handle):
        count = wintypes.ULONG()
        function(handle, None, 0, ctypes.byref(count))
        if not count.value:
            return []
        values = (wintypes.ULONG * count.value)()
        if not function(handle, values, count.value, ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return list(values)

    size = wintypes.ULONG()
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), process, 0)
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetSystemCpuSetInformation(buffer, size, ctypes.byref(size), process, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    records = []
    offset = 0
    while offset < size.value:
        length, kind = struct.unpack_from("II", buffer.raw, offset)
        if length < 8 or offset + length > size.value:
            raise RuntimeError("Invalid Windows CPU set record")
        if kind == 0 and length >= 32:
            cpu_id = struct.unpack_from("I", buffer.raw, offset + 8)[0]
            efficiency, flags = buffer.raw[offset + 18:offset + 20]
            if not flags & 2 or flags & 4:  # not allocated elsewhere
                records.append((cpu_id, efficiency))
        offset += length
    previous = selected(kernel.GetThreadSelectedCpuSets, thread)
    allowed = previous or selected(kernel.GetProcessDefaultCpuSets, process)
    if allowed:
        records = [record for record in records if record[0] in allowed]
    if len({item[1] for item in records}) < 2:
        yield
        return
    highest = max(item[1] for item in records)
    ids = [cpu_id for cpu_id, efficiency in records if efficiency == highest]
    values = (wintypes.ULONG * len(ids))(*ids)
    if not kernel.SetThreadSelectedCpuSets(thread, values, len(ids)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        old = (wintypes.ULONG * len(previous))(*previous) if previous else None
        if not kernel.SetThreadSelectedCpuSets(thread, old, len(previous)):
            raise ctypes.WinError(ctypes.get_last_error())
