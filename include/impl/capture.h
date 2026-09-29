#ifndef FAKE_CUDA_IMPL_CAPTURE_H
#define FAKE_CUDA_IMPL_CAPTURE_H

#include "graph.h"
#include <thread>

namespace fake_cuda::detail {
class Capture {
public:
    Capture(Key origin, CUgraph graph, CUstreamCaptureMode mode)
        : origin(origin), graph_(graph), mode_(mode), owner_(std::this_thread::get_id()) {
        lanes.emplace(origin, 0);
    }
    bool owned_by_thread() const { return owner_ == std::this_thread::get_id(); }
    CUstreamCaptureMode mode() const { return mode_; }
    bool invalidated() const { return invalidated_; }
    void invalidate() { invalidated_ = true; }
    CUgraph graph() const noexcept { return graph_; }
    const std::vector<GraphNode> &graph_nodes() const noexcept { return nodes; }
    bool contains(Key key) const;
    const CUgraphNode *dependency(Key key) const;
    const CUgraphEdgeData *dependency_edge(Key key) const;
    bool matches_origin(Key key) const;
    bool append(Key key, Node node, CUgraphNode handle);
    bool joined() const;
    Graph finish(CUcontext context) const;

private:
    Key origin;
    CUgraph graph_;
    CUstreamCaptureMode mode_;
    std::thread::id owner_;
    bool invalidated_ = false;
    std::vector<GraphNode> nodes;
    std::map<Key, size_t> lanes;
    std::map<Key, size_t> last;
    std::map<Key, CUgraphNode> last_handles;
    CUgraphEdgeData edge_{};
    std::map<CUevent, size_t> recorded;
};
} // namespace fake_cuda::detail

#endif
