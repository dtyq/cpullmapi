
APP := cpullmapi
GO_TAGS ?= with_audio with_image
# CrispASR statically links system opencore-amr as a PRIVATE dependency, so the
# final Go link has to name those libraries explicitly.
GO_LDFLAGS ?= -lopencore-amrnb -lopencore-amrwb
CRISPASR_DIR := third_party/CrispASR
CRISPASR_BUILD_DIR := $(CRISPASR_DIR)/build_go
LLAMACPP_DIR := third_party/llama.cpp
LLAMACPP_BUILD_DIR := $(LLAMACPP_DIR)/build_go

CRISPASR_INCLUDE_PATH := $(abspath $(CRISPASR_DIR))/include:$(abspath $(CRISPASR_DIR))/ggml/include
CRISPASR_LIBRARY_PATH := $(abspath $(CRISPASR_BUILD_DIR))/src:$(abspath $(CRISPASR_BUILD_DIR))/glint:$(abspath $(CRISPASR_BUILD_DIR))/crisp_audio:$(abspath $(CRISPASR_BUILD_DIR))/crisp_lid:$(abspath $(CRISPASR_BUILD_DIR))/crisp_punc:$(abspath $(CRISPASR_BUILD_DIR))/crisp_truecase:$(abspath $(CRISPASR_BUILD_DIR))/ggml/src:$(abspath $(CRISPASR_BUILD_DIR))/_deps/crispasr_ogg-build:$(abspath $(CRISPASR_BUILD_DIR))/_deps/crispasr_opus-build:$(abspath $(CRISPASR_BUILD_DIR))/_deps/crispasr_opencore_amr-build

# llamacppcgo resolves libllama and libmtmd with dlopen at runtime, so cgo only
# needs their headers.
LLAMACPP_CGO_CFLAGS := -I$(abspath $(LLAMACPP_DIR))/include -I$(abspath $(LLAMACPP_DIR))/ggml/include -I$(abspath $(LLAMACPP_DIR))/tools/mtmd

GO_TEST_ARGS ?= -v
GO_ENV := CGO_ENABLED=1 CGO_CFLAGS='$(LLAMACPP_CGO_CFLAGS)' C_INCLUDE_PATH=$(CRISPASR_INCLUDE_PATH) LIBRARY_PATH=$(CRISPASR_LIBRARY_PATH) CGO_LDFLAGS='$(GO_LDFLAGS)'

.PHONY: all build test download go-deps crispasr-whisper llamacpp clean

all: build

build: go-deps crispasr-whisper llamacpp
	@mkdir -p $(dir $(APP))
	@$(GO_ENV) go build -tags '$(GO_TAGS)' -o $(APP) ./cmd

test: go-deps crispasr-whisper llamacpp
	@$(GO_ENV) go test -tags '$(GO_TAGS)' $(GO_TEST_ARGS) ./...

# 用法：make download ARGS="onnx-mvanet silero-vad-v5"，不带 ARGS 就全下。
download:
	@$(GO_ENV) go run ./cmd download $(ARGS)

go-deps:
	@go mod download

crispasr-whisper:
	@cmake -S $(CRISPASR_DIR) -B $(CRISPASR_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DCRISPASR_WITH_ESPEAK_NG=OFF \
		-DCRISPASR_MEL_BLAS=OFF \
		-DCRISPASR_OPUS_FETCH=ON
	@cmake --build $(CRISPASR_BUILD_DIR) --target crispasr-lib

llamacpp:
	@cmake -S $(LLAMACPP_DIR) -B $(LLAMACPP_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=ON \
		-DLLAMA_CURL=OFF
	@cmake --build $(LLAMACPP_BUILD_DIR) --target llama mtmd

clean:
	@rm -f $(APP)
