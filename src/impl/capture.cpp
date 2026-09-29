#include "impl/capture.h"

namespace fake_cuda::detail {

bool Capture::contains(Key key) const { return lanes.contains(key); }
bool Capture::matches_origin(Key key) const {
    return origin.stream == key.stream && origin.thread == key.thread;
}

const CUgraphNode *Capture::dependency(Key key) const {
    auto it = last_handles.find(key);
    return it == last_handles.end() ? nullptr : &it->second;
}
const CUgraphEdgeData *Capture::dependency_edge(Key key) const {
    return dependency(key) ? &edge_ : nullptr;
}
bool Capture::append(Key key, Node node, CUgraphNode handle) {
    auto lane = lanes.find(key);
    auto recorded_event = recorded.find(node.event);
    bool imported = false;
    // Waiting on a captured record imports this stream into the capture.
    if (lane == lanes.end() && node.kind == Kind::wait && recorded_event != recorded.end()) {
        lane = lanes.emplace(key, lanes.size()).first;
        imported = true;
    }
    if (lane == lanes.end()) return false;
    size_t previous = no_node;
    if (auto it = last.find(key); it != last.end()) previous = it->second;
    size_t dependency = node.kind == Kind::wait && recorded_event != recorded.end()
                        ? recorded_event->second : no_node;
    const size_t index = nodes.size();
    // Reserve map entries before updating any existing dependencies. Roll back
    // newly inserted entries if allocation fails, so capture can be retried.
    bool new_tail = false, new_handle = false, new_record = false;
    try {
        auto [tail, inserted_tail] = last.try_emplace(key, no_node);
        new_tail = inserted_tail;
        auto [stream_handle, inserted_handle] = last_handles.try_emplace(key, nullptr);
        new_handle = inserted_handle;
        if (node.kind == Kind::record)
            new_record = recorded.try_emplace(node.event, no_node).second;
        nodes.push_back({node, lane->second, previous, dependency, handle});
        tail->second = index;
        stream_handle->second = handle;
        if (node.kind == Kind::record) recorded.find(node.event)->second = index;
        return true;
    } catch (...) {
        if (new_record) recorded.erase(node.event);
        if (new_handle) last_handles.erase(key);
        if (new_tail) last.erase(key);
        if (imported) lanes.erase(lane);
        throw;
    }
}

bool Capture::joined() const {
    std::vector<bool> reachable(nodes.size());
    std::vector<size_t> todo;
    if (auto it = last.find(origin); it != last.end()) todo.push_back(it->second);
    while (!todo.empty()) {
        size_t index = todo.back();
        todo.pop_back();
        if (reachable[index]) continue;
        reachable[index] = true;
        const GraphNode &node = nodes[index];
        if (node.previous != no_node) todo.push_back(node.previous);
        if (node.event_dependency != no_node) todo.push_back(node.event_dependency);
    }
    for (const auto &[lane, index] : last)
        if (!reachable[index]) return false;
    return true;
}
Graph Capture::finish(CUcontext context) const {
    return Graph(context, nodes, lanes.size());
}
} // namespace fake_cuda::detail
