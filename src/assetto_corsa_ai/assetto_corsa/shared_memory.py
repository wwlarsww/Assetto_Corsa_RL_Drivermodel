"""Open existing simulator mappings without creating competing memory."""

import ctypes
from ctypes import wintypes
import mmap
import os


def open_readonly_mapping(name: str, length: int) -> mmap.mmap:
    if os.name != "nt":
        raise OSError("Assetto Corsa shared memory is only available on Windows.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_mapping = kernel32.OpenFileMappingW
    open_mapping.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    open_mapping.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    handle = open_mapping(0x0004, False, name)  # FILE_MAP_READ
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Hold the existing mapping until mmap owns its handle. Creating an
        # absent read-only mapping can prevent AC's writer from starting.
        return mmap.mmap(-1, length, tagname=name, access=mmap.ACCESS_READ)
    finally:
        close_handle(handle)
