
ARG DEBIAN_VERSION=trixie
ARG DOCKERHUB_LIBRARY_IMAGE_PREFIX=public.ecr.aws/docker/library/

FROM ${DOCKERHUB_LIBRARY_IMAGE_PREFIX}golang:${DEBIAN_VERSION} AS builder

ARG TARGETARCH

ARG DEBIAN_VERSION=trixie
ARG DEBIAN_APT_MIRROR=mirrors.aliyun.com
RUN --mount=type=cache,id=apt-debian-${DEBIAN_VERSION}-${TARGETARCH}-builder,target=/var/cache/apt \
    --mount=type=cache,id=apt-lists-debian-${DEBIAN_VERSION}-${TARGETARCH}-builder,target=/var/lib/apt/lists \
    # setup apt mirror follow deb-822 format
    sed -i.bak "s/deb.debian.org/${DEBIAN_APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources && \
    # debian的docker会自动删除下载的包 导致缓存失效 因此修改一下
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        pkg-config \
        libvips-dev \
        libopencore-amrnb-dev \
        libopencore-amrwb-dev

COPY . /app

WORKDIR /app

# 两个原生依赖的构建目录挂成 cache，增量编译跨构建复用。cache mount 不进镜像层，
# 所以产物要在同一条 RUN 里拷到 /out。
RUN --mount=type=cache,id=crispasr-build-${DEBIAN_VERSION}-${TARGETARCH},target=/app/third_party/CrispASR/build_go \
    cmake -S third_party/CrispASR -B third_party/CrispASR/build_go \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DCRISPASR_WITH_ESPEAK_NG=OFF \
        -DCRISPASR_MEL_BLAS=OFF \
        -DCRISPASR_OPUS_FETCH=ON && \
    cmake --build third_party/CrispASR/build_go --target crispasr-lib -j "$(nproc)"

RUN --mount=type=cache,id=llamacpp-build-${DEBIAN_VERSION}-${TARGETARCH},target=/app/third_party/llama.cpp/build_go \
    cmake -S third_party/llama.cpp -B third_party/llama.cpp/build_go \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_CURL=OFF && \
    cmake --build third_party/llama.cpp/build_go --target llama mtmd -j "$(nproc)" && \
    mkdir -p /out/libs/llama && \
    cp -a third_party/llama.cpp/build_go/bin/libllama.so* third_party/llama.cpp/build_go/bin/libmtmd.so* /out/libs/llama/

ARG GOPROXY=https://goproxy.cn,direct
# llamacppcgo 只是 dlopen 那两个 so，编译期要的是 llama.cpp 的头文件，所以 CGO_CFLAGS 得带上。
RUN --mount=type=cache,id=go-build-${DEBIAN_VERSION}-${TARGETARCH},target=/root/.cache/go-build \
    --mount=type=cache,id=go-mod-${DEBIAN_VERSION}-${TARGETARCH},target=/go/pkg/mod \
    --mount=type=cache,id=crispasr-build-${DEBIAN_VERSION}-${TARGETARCH},target=/app/third_party/CrispASR/build_go \
    GOPROXY=${GOPROXY} \
    CGO_ENABLED=1 \
    CGO_CFLAGS="-I/app/third_party/llama.cpp/include -I/app/third_party/llama.cpp/ggml/include -I/app/third_party/llama.cpp/tools/mtmd" \
    CGO_LDFLAGS="-lopencore-amrnb -lopencore-amrwb" \
    go build -tags 'with_audio with_image' -o /out/cpullmapi ./cmd && \
    # TODO: build these SOs for ascend NPU
    # sherpa 的 so 只有 go 模块缓存里有，顺手拷出来
    case "$(go env GOARCH)" in \
        amd64) _sherpa_triple=x86_64-unknown-linux-gnu ;; \
        arm64) _sherpa_triple=aarch64-unknown-linux-gnu ;; \
        arm)   _sherpa_triple=arm-unknown-linux-gnueabihf ;; \
        *) echo "sherpa-onnx ships no $(go env GOARCH) prebuilt" >&2; exit 1 ;; \
    esac && \
    mkdir -p /out/libs/sherpa && \
    cp -a "$(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-linux@v1.13.2/lib/${_sherpa_triple}"/*.so /out/libs/sherpa/

ARG DEBIAN_VERSION=trixie
ARG DOCKERHUB_LIBRARY_IMAGE_PREFIX=public.ecr.aws/docker/library/
FROM ${DOCKERHUB_LIBRARY_IMAGE_PREFIX}debian:${DEBIAN_VERSION}-slim AS runner

ARG TARGETARCH

ARG DEBIAN_VERSION=trixie
ARG DEBIAN_APT_MIRROR=mirrors.aliyun.com
RUN --mount=type=cache,id=apt-debian-${DEBIAN_VERSION}-${TARGETARCH}-runner,target=/var/cache/apt \
    --mount=type=cache,id=apt-lists-debian-${DEBIAN_VERSION}-${TARGETARCH}-runner,target=/var/lib/apt/lists \
    # setup apt mirror follow deb-822 format
    sed -i.bak "s/deb.debian.org/${DEBIAN_APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources && \
    # debian的docker会自动删除下载的包 导致缓存失效 因此修改一下
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libvips \
        ca-certificates \
        # CrispASR 把 opencore-amr 当私有依赖链进二进制，运行时还得有这两个
        libopencore-amrnb0 \
        libopencore-amrwb0 \
        # crispasr 用 -fopenmp 编译
        libgomp1

COPY --from=builder /out/cpullmapi /usr/local/bin/cpullmapi
COPY --from=builder /out/libs/llama /app/libs/llama
COPY --from=builder /out/libs/sherpa /app/libs/sherpa
# tokenHash 是密钥，不能烧进镜像，所以这里只放样例，真正的 config.yml 运行期挂进来。
COPY config.yml.example /app/config.yml.example

# 工作目录就是仓库根的样子，config.yml 里 ./models/... 和 ./libs/... 那套路径直接对得上。
WORKDIR /app
# sherpa 的 so 是二进制的 DT_NEEDED，进程一启动就要解析，跟跑哪个子命令无关，所以这行
# 必须在下面调用 cpullmapi 之前。
ENV LD_LIBRARY_PATH=/app/libs/sherpa

# 模型下到 cache mount 里，跨构建保留，所以重建不会重下：下载器认 /models.cache 里已有的
# 文件，只会补齐缺的。cache mount 不进镜像层也不能跨 RUN，选中的东西要在同一条 RUN 里
# stage 进 /app。
#
# MODELS 用 download 那套资产名写法（onnx-mvanet、silero-vad-v5，也能带 @版本:标签），
# all 是全要，空是一个都不要。
ARG MODELS=all
ARG MODELS_SOURCE=huggingface
ARG HF_ENDPOINT=https://huggingface.co
ARG MS_ENDPOINT=https://www.modelscope.cn
RUN --mount=type=cache,id=models-${DEBIAN_VERSION}-${TARGETARCH},target=/models.cache \
    --mount=type=cache,id=model-libs-${DEBIAN_VERSION}-${TARGETARCH},target=/libs.cache \
    set -e; \
    mkdir -p /app/models; \
    if [ -z "${MODELS}" ]; then \
        echo "MODELS is empty, the image ships no models"; \
    else \
        if [ "${MODELS}" = "all" ]; then set --; else set -- ${MODELS}; fi; \
        cpullmapi download \
            --source "${MODELS_SOURCE}" \
            --hf-endpoint "${HF_ENDPOINT}" \
            --ms-endpoint "${MS_ENDPOINT}" \
            --models-dir /models.cache \
            --libs-dir /libs.cache \
            --stage-to /app \
            "$@"; \
    fi

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/cpullmapi"]
CMD ["--config", "./config.yml"]
