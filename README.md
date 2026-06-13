# A tree-based model compiler based on MLIR
## How to build
```bash
git clone https://github.com/njru8cjo/tree-house
mkdir build && cd build
cmake ..
cmake --build .
```

## Dump LLVM IR code
```bash
./build/bin/frontend --dump
```

## Pre-request
MLIR build with llvm-project 22.0.0git
https://github.com/llvm/llvm-project/tree/3d38a92c8a835f5f4296c6581f6c13f60d38ba3a

## Paper
https://dl.acm.org/doi/abs/10.1145/3704727
