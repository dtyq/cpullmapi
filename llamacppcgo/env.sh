#!/bin/bash

fileDir=$(dirname $(realpath "$0"))

LLAMA_SRC_DIR=$(realpath "$fileDir/../../llama.cpp")
LLAMA_BUILD_DIR=$(realpath "$fileDir/../../llama.cpp/build")

INCLUDES=(
    -I${LLAMA_SRC_DIR}/include
    -I${LLAMA_SRC_DIR}/ggml/include
    -I${LLAMA_SRC_DIR}/tools/mtmd
)

# LIBS=(
#     -Wl,--start-group
#     -L${LLAMA_BUILD_DIR}/bin
#     -lmtmd
#     -lllama
#     -lggml-base
#     -lggml
#     -Wl,--end-group
# )

echo set CGO_CFLAGS="${INCLUDES[@]} -g"
export CGO_CFLAGS="${INCLUDES[@]} -g"
# export CGO_LDFLAGS="${LIBS[@]}"
