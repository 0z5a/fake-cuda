#ifndef FAKE_CUDA_IMPL_GRAPH_H
#define FAKE_CUDA_IMPL_GRAPH_H

#include "virtual_types.h"

#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fake_cuda::detail {
class Capture;
class Scheduler;

inline constexpr size_t no_node = std::numeric_limits<size_t>::max();
struct GraphNode {
    Node node;
    size_t lane, previous = no_node, event_dependency = no_node;
    CUgraphNode handle = nullptr;
};

// Immutable graph definition; executable handles retain their own shared snapshot.
class Graph {
public:
    Graph(CUcontext context, std::vector<GraphNode> nodes, size_t lanes) noexcept;
    CUcontext context() const noexcept { return context_; }
    const std::vector<GraphNode> &nodes() const noexcept { return nodes_; }
    CUresult supports(unsigned long long flags) const noexcept;
    CUresult launch(Scheduler &scheduler, Key key, OpPtr &completion) const;

private:
    CUcontext context_;
    std::vector<GraphNode> nodes_;
    size_t lanes_;
};

// Owns capture sessions, finished definitions and executable handles. All calls
// are made under Scheduler::mutex; graph executions share immutable definitions.
class GraphManager {
public:
    GraphManager();
    ~GraphManager();
    GraphManager(const GraphManager &) = delete;
    GraphManager &operator=(const GraphManager &) = delete;

    bool capturing(Key key) const;
    CUresult append(Key key, Node node, CUcontext event_context, CUgraphNode handle, bool &captured);
    CUresult begin_capture(Scheduler &scheduler, Key key, CUstreamCaptureMode mode);
    CUresult invalidate(Key key);
    CUresult invalidate_context(CUcontext context);
    CUresult check_unsafe_call();
    CUresult end_capture(Key key, CUgraph *graph);
    void capture_info(Key key, CUstreamCaptureStatus *status, cuuint64_t *id,
                      CUgraph *graph, const CUgraphNode **dependencies,
                      const CUgraphEdgeData **edges, size_t *count) const;
    CUresult get_nodes(CUcontext context, CUgraph graph, CUgraphNode *nodes, size_t *count) const;
    CUresult instantiate(Scheduler &scheduler, CUcontext context, CUgraph graph,
                         unsigned long long flags, CUgraphExec *exec);
    CUresult launch(Scheduler &scheduler, CUgraphExec exec, Key key);
    CUresult destroy(CUcontext context, CUgraph graph);
    CUresult destroy_exec(CUcontext context, CUgraphExec exec);
    void retire_stream(CUcontext context, Key key);
    void retire_context(CUcontext context);
    bool active(CUcontext context) const { return captures_.contains(context); }

private:
    std::map<CUcontext, std::unique_ptr<Capture>> captures_;
    std::unordered_map<CUgraph, std::shared_ptr<const Graph>> graphs_;
    struct Executable {
        std::shared_ptr<const Graph> graph;
        OpPtr completion;
    };
    std::unordered_map<CUgraphExec, Executable> executables_;
};
} // namespace fake_cuda::detail

#endif
