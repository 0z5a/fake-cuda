#include "impl/private.h"
#include "impl/core.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <new>
#include <pthread.h>
#include <unistd.h>

/* CUDA's private UUIDs, table layouts, seed and substitution table below are
 * protocol data described by ZLUDA dark_api/src/lib.rs (vosen/ZLUDA).
 * ZLUDA is licensed under MIT (and Apache-2.0); this file uses its MIT grant.
 *
 * Copyright (c) the ZLUDA contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * The implementation here is independently written against that protocol;
 * these interfaces are undocumented and scoped to the observed CUDA 13 ABI.
 */

namespace {
static_assert(sizeof(void *) == 8 && sizeof(CUuuid) == 16, "CUDA 13 private ABI requires 64-bit pointers");

using Slot = const void *;
// Confirmed in ZLUDA dark_api: callback(context, key, value), extern "system".
using Destructor = void (CUDAAPI *)(CUcontext, void *, void *);
struct LocalEntry {
    CUcontext context;
    void *key;
    void *value;
    Destructor destructor;
};

class ContextLocalStorage {
    std::mutex mutex_;
    std::list<LocalEntry> entries_;

    static CUresult resolve_context(CUcontext &context) {
        if (!context) {
            CUresult result = core_context_current(&context);
            if (result != CUDA_SUCCESS) return result;
        }
        if (!context) return CUDA_ERROR_INVALID_CONTEXT;
        // Reject retired/foreign handles without changing the calling thread's context.
        unsigned int ignored = 0;
        return core_context_version(context, &ignored);
    }

public:
    CUresult put(CUcontext context, void *key, void *value, Destructor destructor) {
        try {
            std::lock_guard lock(mutex_);
            // Validate while holding the CLS lock: teardown detaches entries only
            // after retiring the context, and never holds the registry lock here.
            CUresult result = resolve_context(context);
            if (result != CUDA_SUCCESS) return result;
            for (auto &entry : entries_) {
                if (entry.context == context && entry.key == key) {
                    entry.value = value;
                    entry.destructor = destructor;
                    return CUDA_SUCCESS;
                }
            }
            entries_.push_back({context, key, value, destructor});
            return CUDA_SUCCESS;
        } catch (const std::bad_alloc &) { return CUDA_ERROR_OUT_OF_MEMORY; }
    }

    CUresult erase(CUcontext context, void *key) {
        std::lock_guard lock(mutex_);
        CUresult result = resolve_context(context);
        if (result != CUDA_SUCCESS) return result;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->context == context && it->key == key) {
                entries_.erase(it);
                break;
            }
        }
        return CUDA_SUCCESS;
    }

    CUresult get(void **value, CUcontext context, void *key) {
        if (!value) return CUDA_ERROR_INVALID_VALUE;
        *value = nullptr;
        std::lock_guard lock(mutex_);
        CUresult result = resolve_context(context);
        if (result != CUDA_SUCCESS) return result;
        for (const auto &entry : entries_) {
            if (entry.context == context && entry.key == key) {
                *value = entry.value;
                return CUDA_SUCCESS;
            }
        }
        return CUDA_ERROR_INVALID_HANDLE;
    }

    void context_destroyed(CUcontext context) {
        if (!context) return;
        std::list<LocalEntry> removed;
        {
            std::lock_guard lock(mutex_);
            for (auto it = entries_.begin(); it != entries_.end();) {
                if (it->context == context) {
                    auto entry = it++;
                    removed.splice(removed.end(), entries_, entry);
                } else {
                    ++it;
                }
            }
        }
        // Detached nodes are owned by removed until callbacks finish; no core
        // or CLS lock is held while re-entrant callbacks use the Driver.
        for (const auto &entry : removed) {
            if (entry.destructor) {
                try {
                    entry.destructor(entry.context, entry.key, entry.value);
                } catch (...) {
                    // A foreign callback must not unwind through this C ABI or skip
                    // the remaining cleanup callbacks.
                    if (std::getenv("FAKE_CUDA_TRACE"))
                        std::fprintf(stderr, "fake-cuda: CLS destructor threw an exception\n");
                }
            }
        }
    }
};

ContextLocalStorage local_storage;
CUresult CUDAAPI local_put(CUcontext context, void *key, void *value, Destructor destructor) {
    return local_storage.put(context, key, value, destructor);
}
CUresult CUDAAPI local_delete(CUcontext context, void *key) {
    return local_storage.erase(context, key);
}
CUresult CUDAAPI local_get(void **value, CUcontext context, void *key) {
    return local_storage.get(value, context, key);
}

CUresult CUDAAPI no_cubin(CUmodule *, const void *) { return CUDA_ERROR_NOT_SUPPORTED; }
CUresult CUDAAPI no_cubin_ext(CUmodule *, const void *, void *, void *, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI no_cubin_header(const void *, CUmodule *, void *, void *, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI primary_context(CUcontext *out, CUdevice device) {
    return core_primary_get(out, device);
}
CUresult CUDAAPI load_compilers() {
    // Optional loader handshake; actual module/compilation operations are unsupported.
    return CUDA_SUCCESS;
}
std::uint32_t callback_words[1024] = {};
std::uint32_t callback_small[14] = {};
void CUDAAPI callback_large(void **ptr, size_t *size) {
    if (ptr) *ptr = callback_words;
    if (size) *size = 1024;
}
void CUDAAPI callback_short(void **ptr, size_t *size) {
    if (ptr) *ptr = callback_small;
    if (size) *size = 14;
}
CUresult CUDAAPI context_check(CUcontext, std::uint32_t *status, const void **extra) {
    if (status) *status = 0;
    if (extra) *extra = nullptr;
    return CUDA_SUCCESS;
}
std::uint32_t CUDAAPI context_status() { return 0; }

// The substitution constants and initial digest are protocol data from ZLUDA.
constexpr std::uint8_t substitution[256] = {
    0x29,0x2e,0x43,0xc9,0xa2,0xd8,0x7c,0x01,0x3d,0x36,0x54,0xa1,0xec,0xf0,0x06,0x13,
    0x62,0xa7,0x05,0xf3,0xc0,0xc7,0x73,0x8c,0x98,0x93,0x2b,0xd9,0xbc,0x4c,0x82,0xca,
    0x1e,0x9b,0x57,0x3c,0xfd,0xd4,0xe0,0x16,0x67,0x42,0x6f,0x18,0x8a,0x17,0xe5,0x12,
    0xbe,0x4e,0xc4,0xd6,0xda,0x9e,0xde,0x49,0xa0,0xfb,0xf5,0x8e,0xbb,0x2f,0xee,0x7a,
    0xa9,0x68,0x79,0x91,0x15,0xb2,0x07,0x3f,0x94,0xc2,0x10,0x89,0x0b,0x22,0x5f,0x21,
    0x80,0x7f,0x5d,0x9a,0x5a,0x90,0x32,0x27,0x35,0x3e,0xcc,0xe7,0xbf,0xf7,0x97,0x03,
    0xff,0x19,0x30,0xb3,0x48,0xa5,0xb5,0xd1,0xd7,0x5e,0x92,0x2a,0xac,0x56,0xaa,0xc6,
    0x4f,0xb8,0x38,0xd2,0x96,0xa4,0x7d,0xb6,0x76,0xfc,0x6b,0xe2,0x9c,0x74,0x04,0xf1,
    0x45,0x9d,0x70,0x59,0x64,0x71,0x87,0x20,0x86,0x5b,0xcf,0x65,0xe6,0x2d,0xa8,0x02,
    0x1b,0x60,0x25,0xad,0xae,0xb0,0xb9,0xf6,0x1c,0x46,0x61,0x69,0x34,0x40,0x7e,0x0f,
    0x55,0x47,0xa3,0x23,0xdd,0x51,0xaf,0x3a,0xc3,0x5c,0xf9,0xce,0xba,0xc5,0xea,0x26,
    0x2c,0x53,0x0d,0x6e,0x85,0x28,0x84,0x09,0xd3,0xdf,0xcd,0xf4,0x41,0x81,0x4d,0x52,
    0x6a,0xdc,0x37,0xc8,0x6c,0xc1,0xab,0xfa,0x24,0xe1,0x7b,0x08,0x0c,0xbd,0xb1,0x4a,
    0x78,0x88,0x95,0x8b,0xe3,0x63,0xe8,0x6d,0xe9,0xcb,0xd5,0xfe,0x3b,0x00,0x1d,0x39,
    0xf2,0xef,0xb7,0x0e,0x66,0x58,0xd0,0xe4,0xa6,0x77,0x72,0xf8,0xeb,0x75,0x4b,0x0a,
    0x31,0x44,0x50,0xb4,0x8f,0xed,0x1f,0x1a,0xdb,0x99,0x8d,0x33,0x9f,0x11,0x83,0x14
};
constexpr std::uint8_t initial_digest[16] = {
    0x14,0x6a,0xdd,0xae,0x53,0xa9,0xa7,0x52,0xaa,0x08,0x41,0x36,0x0b,0xf5,0x5a,0x9f
};

class IntegrityMixer {
    std::array<std::uint8_t, 66> data_{};
    void add(std::uint8_t byte) {
        const unsigned offset = data_[64];
        data_[16 + offset] = byte;
        data_[32 + offset] = data_[offset] ^ byte;
        const std::uint8_t mixed = substitution[byte ^ data_[65]] ^ data_[48 + offset];
        data_[48 + offset] = data_[65] = mixed;
        data_[64] = (offset + 1) & 15;
        if (data_[64]) return;
        std::uint8_t carry = 0x29;
        for (unsigned round = 0; round != 18; ++round) {
            carry ^= data_[0];
            data_[0] = carry;
            for (unsigned i = 1; i != 48; ++i) {
                carry = data_[i] ^ substitution[carry];
                data_[i] = carry;
            }
            carry = substitution[static_cast<std::uint8_t>(carry + round)];
        }
    }
public:
    void absorb(const void *input, size_t length, std::uint8_t mask = 0) {
        auto bytes = static_cast<const std::uint8_t *>(input);
        for (size_t i = 0; i < length; ++i) add(bytes[i] ^ mask);
    }
    void finish(std::uint64_t out[2]) {
        const unsigned padding = 16 - data_[64];
        for (unsigned i = 0; i < padding; ++i) add(static_cast<std::uint8_t>(padding));
        for (unsigned i = 48; i < 64; ++i) add(data_[i]);
        std::memcpy(out, data_.data(), 16);
    }
    void restart_second_pass() {
        std::memset(data_.data(), 0, 16);
        std::memset(data_.data() + 48, 0, 18);
    }
};

struct CheckInput {
    std::uint32_t driver_version, version, process, thread;
    const void *cudart_table, *integrity_table, *function;
    std::uint64_t unix_seconds;
};
struct DeviceInfo {
    CUuuid uuid;
    std::int32_t domain, bus, device;
};
static_assert(sizeof(CheckInput) == 48 && sizeof(DeviceInfo) == 28, "unexpected integrity ABI padding");

extern const Slot cudart_table[13];
extern const Slot integrity_table[3];
CUresult CUDAAPI integrity_check(std::uint32_t version, std::uint64_t seconds, std::uint64_t output[2]) {
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    if (version % 10 == 0 || version % 10 == 1) {
        output[0] = version % 10 == 0 ? UINT64_C(0x3341181c03cb675c) : UINT64_C(0x1841181c03cb675c);
        output[1] = UINT64_C(0x8ed383aa1f4cd1e8);
        return CUDA_SUCCESS;
    }
    int driver_version = 0;
    CUresult result = core_version(&driver_version);
    if (result != CUDA_SUCCESS) return result;
    int count = 0;
    result = core_count(&count);
    if (result != CUDA_SUCCESS) return result;
    if (count < 0) return CUDA_ERROR_UNKNOWN;

    const pthread_t thread = pthread_self();
    static_assert(sizeof(thread) <= sizeof(std::uintptr_t), "unsupported pthread_t ABI");
    std::uintptr_t thread_bits = 0;
    std::memcpy(&thread_bits, &thread, sizeof(thread));
    CheckInput input{};
    input.driver_version = static_cast<std::uint32_t>(driver_version);
    input.version = version;
    input.process = static_cast<std::uint32_t>(getpid());
    input.thread = static_cast<std::uint32_t>(thread_bits);
    input.cudart_table = cudart_table;
    input.integrity_table = integrity_table;
    input.function = reinterpret_cast<const void *>(&integrity_check);
    input.unix_seconds = seconds;

    IntegrityMixer mixer;
    mixer.absorb(initial_digest, sizeof(initial_digest), 0x36);
    mixer.absorb(&input, sizeof(input));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        CUdevice device = 0;
        DeviceInfo info{};
        if ((result = core_device(&device, ordinal)) != CUDA_SUCCESS ||
            (result = core_uuid(&info.uuid, device)) != CUDA_SUCCESS ||
            (result = core_attribute(&info.domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device)) != CUDA_SUCCESS ||
            (result = core_attribute(&info.bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device)) != CUDA_SUCCESS ||
            (result = core_attribute(&info.device, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device)) != CUDA_SUCCESS)
            return result;
        mixer.absorb(&info, sizeof(info));
    }
    std::uint64_t intermediate[2];
    mixer.finish(intermediate);
    mixer.restart_second_pass();
    mixer.absorb(initial_digest, sizeof(initial_digest), 0x5c);
    mixer.absorb(intermediate, sizeof(intermediate));
    mixer.finish(output);
    return CUDA_SUCCESS;
}

// Slots are pointer-sized and remain valid for the entire lifetime of the library.
const Slot cudart_table[13] = {
    reinterpret_cast<Slot>(13 * sizeof(Slot)), reinterpret_cast<Slot>(&no_cubin),
    reinterpret_cast<Slot>(&primary_context), nullptr, nullptr, nullptr,
    reinterpret_cast<Slot>(&no_cubin_ext), nullptr, reinterpret_cast<Slot>(&no_cubin_header),
    nullptr, nullptr, nullptr, reinterpret_cast<Slot>(&load_compilers)
};
const Slot tools_table[4] = {reinterpret_cast<Slot>(4 * sizeof(Slot))};
const Slot callbacks_table[7] = {
    reinterpret_cast<Slot>(7 * sizeof(Slot)), nullptr, reinterpret_cast<Slot>(&callback_large),
    nullptr, nullptr, nullptr, reinterpret_cast<Slot>(&callback_short)
};
const Slot local_table[4] = {
    reinterpret_cast<Slot>(&local_put), reinterpret_cast<Slot>(&local_delete),
    reinterpret_cast<Slot>(&local_get), nullptr
};
const Slot integrity_table[3] = {
    reinterpret_cast<Slot>(3 * sizeof(Slot)), reinterpret_cast<Slot>(&integrity_check), nullptr
};
const Slot checks_table[4] = {
    reinterpret_cast<Slot>(4 * sizeof(Slot)), nullptr,
    reinterpret_cast<Slot>(&context_check), reinterpret_cast<Slot>(&context_status)
};
// Observed CUDA 13 bootstrap table: length only. No unknown callable slots invented.
const Slot bootstrap_table[3] = {reinterpret_cast<Slot>(3 * sizeof(Slot))};

struct Export {
    std::uint8_t uuid[16];
    const Slot *table;
    const char *name;
};
const Export exports[] = {
    {{0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9}, cudart_table, "cudart"},
    {{0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc}, tools_table, "tools-tls"},
    {{0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66}, callbacks_table, "callbacks"},
    {{0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93}, local_table, "context-local-storage"},
    {{0xd4,0x08,0x20,0x55,0xbd,0xe6,0x70,0x4b,0x8d,0x34,0xba,0x12,0x3c,0x66,0xe1,0xf2}, integrity_table, "integrity"},
    {{0x26,0x3e,0x88,0x60,0x7c,0xd2,0x61,0x43,0x92,0xf6,0xbb,0xd5,0x00,0x6d,0xfa,0x7e}, checks_table, "context-checks"},
    {{0xf8,0xcf,0xf9,0x51,0x21,0x46,0x8b,0x4e,0xb9,0xe2,0xfb,0x46,0x9e,0x7c,0x0d,0xd9}, bootstrap_table, "bootstrap"}
};
} // namespace

extern "C" void private_context_destroyed(CUcontext context) {
    local_storage.context_destroyed(context);
}

extern "C" CUresult private_export_table(const void **table, const CUuuid *uuid) {
    if (!table || !uuid) return CUDA_ERROR_INVALID_VALUE;
    *table = nullptr;
    for (const auto &entry : exports) {
        if (std::memcmp(uuid->bytes, entry.uuid, sizeof(entry.uuid)) == 0) {
            *table = entry.table;
            if (std::getenv("FAKE_CUDA_TRACE"))
                std::fprintf(stderr, "fake-cuda: private table %s\n", entry.name);
            return CUDA_SUCCESS;
        }
    }
    if (std::getenv("FAKE_CUDA_TRACE"))
        std::fprintf(stderr, "fake-cuda: unsupported private table %02x%02x%02x%02x\n",
                     static_cast<unsigned char>(uuid->bytes[0]), static_cast<unsigned char>(uuid->bytes[1]),
                     static_cast<unsigned char>(uuid->bytes[2]), static_cast<unsigned char>(uuid->bytes[3]));
    return CUDA_ERROR_NOT_SUPPORTED;
}
