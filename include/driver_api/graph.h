#ifndef FAKE_CUDA_DRIVER_API_GRAPH_H
#define FAKE_CUDA_DRIVER_API_GRAPH_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

CU_EXPORT CUresult CUDAAPI cuGraphGetNodes(CUgraph graph, CUgraphNode *nodes, size_t *count);
CU_EXPORT CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *exec, CUgraph graph,
                                                        unsigned long long flags);
CU_EXPORT CUresult CUDAAPI cuGraphInstantiateWithParams(CUgraphExec *exec, CUgraph graph, CUDA_GRAPH_INSTANTIATE_PARAMS *params);
CU_EXPORT CUresult CUDAAPI cuGraphLaunch(CUgraphExec exec, CUstream stream);
CU_EXPORT CUresult CUDAAPI cuGraphDestroy(CUgraph graph);
CU_EXPORT CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec exec);

#ifdef __cplusplus
}
#endif

#endif /* FAKE_CUDA_DRIVER_API_GRAPH_H */
