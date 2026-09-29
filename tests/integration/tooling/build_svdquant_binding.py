"""Link official Nunchaku operator objects into a small test-only Python module.

Requires the core objects from the pinned source's setup.py build_ext. Does not
compile a replacement kernel or install anything into the Python environment.
"""
import argparse
from pathlib import Path
from torch.utils.cpp_extension import load
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--build', type=Path, required=True)
args = parser.parse_args()
root, build = args.source.resolve(), args.build.resolve()
build.mkdir(exist_ok=True)
source = build / 'binding.cpp'
source.write_text('''#include "ops.h"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def_submodule("ops")
        .def("gemm_w4a4", nunchaku::ops::gemm_w4a4)
        .def("quantize_w4a4_act_fuse_lora", nunchaku::ops::quantize_w4a4_act_fuse_lora);
}
''')
objects = sorted((root/'build/temp.linux-x86_64-cpython-312/src').rglob('*.o'))
objects = [p for p in objects if p.name not in ('FluxModel.o', 'SanaModel.o')]
assert len(objects) == 23, [str(p) for p in objects]
load(name='_C', sources=[str(source)], build_directory=str(build),
     extra_include_paths=[str(root/p) for p in ('nunchaku/csrc','src','third_party/cutlass/include','third_party/json/include','third_party/mio/include','third_party/spdlog/include')],
     extra_cflags=['-std=c++20','-DENABLE_BF16=1','-DBUILD_NUNCHAKU=1','-O2'],
     extra_ldflags=[str(p) for p in objects], with_cuda=True, verbose=True)
