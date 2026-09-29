#ifndef FAKE_CUDA_IMPL_CAPTURE_H
#define FAKE_CUDA_IMPL_CAPTURE_H

#include "graph.h"

namespace fake_cuda::detail {
class Capture {
public:
    Capture(Key origin, CUgraph graph) : origin(origin), graph_(graph) { lanes.emplace(origin, 0); }
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
    std::vector<GraphNode> nodes;
    std::map<Key, size_t> lanes;
    std::map<Key, size_t> last;
    std::map<Key, CUgraphNode> last_handles;
    CUgraphEdgeData edge_{};
    std::map<CUevent, size_t> recorded;
};
} // namespace fake_cuda::detail

#endif
