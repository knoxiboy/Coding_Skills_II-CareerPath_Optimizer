import ctypes
import os
import numpy as np

# Determine the path to the shared library
lib_path = os.path.join(os.path.dirname(__file__), "lib_optimizer.dll")

# Load the library
try:
    _lib = ctypes.CDLL(lib_path)
    _lib_available = True
except Exception as e:
    print(f"Warning: Could not load C++ library: {e}")
    _lib_available = False

if _lib_available:
    # Define greedy_optimize signature
    # int greedy_optimize(int* hours, int* values, int n, int max_hours, int* result_indices)
    _lib.greedy_optimize.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)
    ]
    _lib.greedy_optimize.restype = ctypes.c_int

    # Define dp_optimize signature
    # int dp_optimize(int* hours, int* values, int n, int max_hours, int* result_indices)
    _lib.dp_optimize.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)
    ]
    _lib.dp_optimize.restype = ctypes.c_int

def is_cpp_available():
    return _lib_available

def greedy_optimize_cpp(courses, max_hours):
    if not _lib_available or not courses:
        return [], 0, 0

    n = len(courses)
    hours_arr = (ctypes.c_int * n)(*[int(c['hours']) for c in courses])
    values_arr = (ctypes.c_int * n)(*[int(c['value']) for c in courses])
    result_indices = (ctypes.c_int * n)()

    count = _lib.greedy_optimize(hours_arr, values_arr, n, int(max_hours), result_indices)

    selected = [courses[result_indices[i]] for i in range(count)]
    total_hours = sum(c['hours'] for c in selected)
    total_value = sum(c['value'] for c in selected)

    return selected, total_hours, total_value

def dp_optimize_cpp(courses, max_hours):
    if not _lib_available or not courses:
        return [], 0, 0, []

    n = len(courses)
    hours_arr = (ctypes.c_int * n)(*[int(c['hours']) for c in courses])
    values_arr = (ctypes.c_int * n)(*[int(c['value']) for c in courses])
    result_indices = (ctypes.c_int * n)()

    count = _lib.dp_optimize(hours_arr, values_arr, n, int(max_hours), result_indices)

    selected = [courses[result_indices[i]] for i in range(count)]
    total_hours = sum(c['hours'] for c in selected)
    total_value = sum(c['value'] for c in selected)

    # Note: We don't return the full DP table here as it's massive and usually not needed in UI
    return selected, total_hours, total_value, None
