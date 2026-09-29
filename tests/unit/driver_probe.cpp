#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", #test, __FILE__, __LINE__); exit(1); } } while (0)
#define LOAD(target, name) do { (target) = reinterpret_cast<decltype(target)>(dlsym(lib, name)); CHECK(target != NULL); } while (0)
#define OK(call) CHECK((call) == CUDA_SUCCESS)

int main(int argc, char **argv) {
    CHECK(argc == 2);
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) fprintf(stderr, "dlopen: %s\n", dlerror());
    CHECK(lib != NULL);
    CUresult (*init)(unsigned int), (*count)(int *), (*get)(CUdevice *, int);
    CUresult (*name)(char *, int, CUdevice), (*retain)(CUcontext *, CUdevice);
    CUresult (*total_mem)(size_t *, CUdevice), (*attribute)(int *, CUdevice_attribute, CUdevice);
    CUresult (*current)(CUcontext *), (*set)(CUcontext), (*release)(CUdevice);
    CUresult (*stream_create)(CUstream *, unsigned int), (*stream_ctx)(CUstream, CUcontext *);
    CUresult (*stream_destroy)(CUstream);
    CUresult (*proc)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
    LOAD(init, "cuInit"); LOAD(count, "cuDeviceGetCount"); LOAD(get, "cuDeviceGet");
    LOAD(name, "cuDeviceGetName"); LOAD(retain, "cuDevicePrimaryCtxRetain");
    LOAD(total_mem, "cuDeviceTotalMem_v2"); LOAD(attribute, "cuDeviceGetAttribute");
    LOAD(current, "cuCtxGetCurrent"); LOAD(set, "cuCtxSetCurrent");
    LOAD(release, "cuDevicePrimaryCtxRelease_v2"); LOAD(stream_create, "cuStreamCreate");
    LOAD(stream_ctx, "cuStreamGetCtx"); LOAD(stream_destroy, "cuStreamDestroy_v2");
    LOAD(proc, "cuGetProcAddress_v2");
    OK(init(0));
    int devices = -1; OK(count(&devices)); CHECK(devices == 1);
    CUdevice device = -1; OK(get(&device, 0)); CHECK(get(&device, 1) == CUDA_ERROR_INVALID_DEVICE);
    char label[128]; OK(name(label, sizeof(label), device));
    CHECK(strcmp(label, "NVIDIA GH200 144G HBM3e") == 0);
    size_t bytes = 0; OK(total_mem(&bytes, device)); CHECK(bytes == 153008209920ULL);
    int value = 0;
    OK(attribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)); CHECK(value == 132);
    OK(attribute(&value, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device)); CHECK(value == 62914560);
    OK(attribute(&value, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device)); CHECK(value == 6144);
    CUcontext primary = NULL, found = (CUcontext)1;
    OK(retain(&primary, device)); OK(current(&found)); CHECK(found == NULL);
    OK(set(primary)); OK(current(&found)); CHECK(found == primary);
    CUstream stream = NULL; OK(stream_create(&stream, 0));
    OK(stream_ctx(stream, &found)); CHECK(found == primary);
    void *fn = NULL; CUdriverProcAddressQueryResult status;
    OK(proc("cuDeviceGetCount", &fn, 13000, 0, &status));
    CHECK(fn == (void *)count && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuCtxCreate", &fn, 12080, 0, &status));
    CHECK(fn == dlsym(lib, "cuCtxCreate_v2") && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuCtxCreate", &fn, 13000, 0, &status));
    CHECK(fn == dlsym(lib, "cuCtxCreate_v4") && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuGetProcAddress", &fn, 11030, 0, &status));
    CHECK(fn == dlsym(lib, "cuGetProcAddress") && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuGetProcAddress", &fn, 12000, 0, &status));
    CHECK(fn == dlsym(lib, "cuGetProcAddress_v2") && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuMemAlloc", &fn, 13000, 0, &status));
    CHECK(fn == dlsym(lib, "cuMemAlloc_v2") && status == CU_GET_PROC_ADDRESS_SUCCESS);
    OK(proc("cuUnknownFutureAPI", &fn, 13000, 0, &status));
    CHECK(fn == NULL && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND);
    OK(stream_destroy(stream)); CHECK(stream_ctx(stream, &found) == CUDA_ERROR_INVALID_HANDLE);
    OK(release(device)); OK(current(&found)); CHECK(found == NULL);
    CHECK(set(primary) == CUDA_ERROR_INVALID_CONTEXT);
    CUcontext replacement = NULL;
    OK(retain(&replacement, device));
    CHECK(replacement != primary);
    CHECK(stream_ctx(stream, &found) == CUDA_ERROR_INVALID_HANDLE);
    OK(release(device));
    dlclose(lib);
    puts("PASS: fake Driver discovery, primary context, streams, lookup");
    return 0;
}
