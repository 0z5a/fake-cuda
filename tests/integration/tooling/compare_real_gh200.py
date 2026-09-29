"""Optional host-only check: compare the fake device's reported profile to GPU 0."""
import ctypes as C
import ctypes.util as ctypes_util
import sys


def driver(path):
    library = C.CDLL(path)
    library.cuDeviceGetName.argtypes = [C.c_char_p, C.c_int, C.c_int]
    library.cuDeviceTotalMem_v2.argtypes = [C.POINTER(C.c_size_t), C.c_int]
    library.cuDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    assert library.cuInit(0) == 0
    return library


def profile(library):
    count = C.c_int()
    assert library.cuDeviceGetCount(C.byref(count)) == 0 and count.value >= 1
    device = C.c_int()
    assert library.cuDeviceGet(C.byref(device), 0) == 0
    name = C.create_string_buffer(128)
    assert library.cuDeviceGetName(name, len(name), device) == 0
    size = C.c_size_t()
    assert library.cuDeviceTotalMem_v2(C.byref(size), device) == 0
    attrs = {}
    for index in range(1, 149):
        value = C.c_int()
        result = library.cuDeviceGetAttribute(C.byref(value), index, device)
        attrs[index] = (result, value.value if result == 0 else None)
    return name.value, size.value, attrs


if __name__ == "__main__":
    assert len(sys.argv) == 2, "pass the path to build/libcuda.so.1"
    host_path = ctypes_util.find_library("cuda")
    if not host_path:
        raise SystemExit("A real NVIDIA driver is required for this optional comparison")
    actual = profile(driver(host_path))
    fake = profile(driver(sys.argv[1]))
    assert fake == actual, [
        (key, fake[2][key], actual[2][key])
        for key in fake[2] if fake[2][key] != actual[2][key]
    ]
    print("PASS: fake device 0 matches host GH200 name, capacity, and 148 Driver attributes")
