#ifndef FAKE_CUDA_IMPL_QUEUE_SCHEDULER_H
#define FAKE_CUDA_IMPL_QUEUE_SCHEDULER_H

#include "virtual_types.h"

#include <map>
#include <utility>

namespace fake_cuda::detail {
class QueueScheduler {
public:
    static VirtualClock::duration duration(Kind kind, size_t bytes);

    // Call under Scheduler::mutex; contexts must be registered before submitting work.
    void register_context(CUcontext ctx, CUdevice ordinal);
    CUdevice device_for(CUcontext ctx) const { return devices.at(ctx); }
    Time earliest(Key key, Kind kind, const OpPtr &dependency = {}, Time current = VirtualClock::now()) const;
    OpPtr schedule(Key key, Kind kind, size_t bytes = 0, const OpPtr &dependency = {},
                   Time current = VirtualClock::now());
    // Peer ordinals must be validated by the caller.
    Time earliest_peer(Key key, CUdevice src, CUdevice dst, const OpPtr &dependency = {},
                       Time current = VirtualClock::now()) const;
    OpPtr schedule_peer(Key key, CUdevice src, CUdevice dst, size_t bytes,
                        const OpPtr &dependency = {}, Time current = VirtualClock::now());
    // Include source-context work in context drain and synchronous free deadlines.
    void include_context(CUcontext ctx, Time end);
    void reap(Time now);
    Time stream_end(Key key) const;
    Time context_end(CUcontext ctx) const;
    void retire_stream(Key key) { tails.erase(key); }
    void retire_context(CUcontext ctx);

private:
    Time earliest_on(Key key, const ExecutionQueue *resource_queue, const OpPtr &dependency,
                     Time current) const;
    OpPtr schedule_on(Key key, Kind kind, size_t bytes, const OpPtr &dependency,
                      Time current, ExecutionQueue *resource_queue);
    std::map<CUcontext, CUdevice> devices;

    std::map<Key, OpPtr> tails;
    std::multimap<Time, std::pair<Key, OpPtr>> pending;
    std::map<CUcontext, Time> outstanding;
    std::map<CUcontext, Time> blocking_end;
    std::map<CUcontext, std::weak_ptr<Op>> blocking_last;
};
} // namespace fake_cuda::detail

#endif
