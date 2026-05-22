
ARG DEBIAN_VERSION=trixie
ARG DOCKERHUB_LIBRARY_IMAGE_PREFIX=public.ecr.aws/docker/library/
FROM ${DOCKERHUB_LIBRARY_IMAGE_PREFIX}golang:${DEBIAN_VERSION} AS builder

ARG TARGETARCH

ARG DEBIAN_VERSION=trixie
ARG DEBIAN_APT_MIRROR=mirrors.aliyun.com
RUN --mount=type=cache,id=apt-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/var/cache/apt \
    --mount=type=cache,id=apt-lists-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/var/lib/apt/lists \
    # setup apt mirror follow deb-822 format
    sed -i.bak "s/deb.debian.org/${DEBIAN_APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources && \
    # debian的docker会自动删除下载的包 导致缓存失效 因此修改一下
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libvips-dev

COPY . /app

WORKDIR /app

ARG GOPROXY=https://goproxy.cn,direct
RUN --mount=type=cache,id=go-build-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/root/.cache/go-build \
    --mount=type=cache,id=go-mod-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/go/pkg/mod \
    GOPROXY=${GOPROXY} \
    go build -o cpullmapi ./cmd && \
    # TODO: build these SOs for ascend NPU
    cp /go/pkg/mod/github.com/k2-fsa/sherpa-onnx-go-linux@v1.13.2/lib/aarch64-unknown-linux-gnu/*.so /

ARG DEBIAN_VERSION=trixie
ARG DOCKERHUB_LIBRARY_IMAGE_PREFIX=public.ecr.aws/docker/library/
FROM ${DOCKERHUB_LIBRARY_IMAGE_PREFIX}debian:${DEBIAN_VERSION}-slim AS runner

COPY --from=builder /app/cpullmapi /usr/local/bin/cpullmapi
COPY --from=builder /*.so /usr/local/lib/

ARG DEBIAN_VERSION=trixie
ARG DEBIAN_APT_MIRROR=mirrors.aliyun.com
RUN --mount=type=cache,id=apt-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/var/cache/apt \
    --mount=type=cache,id=apt-lists-debian-${DEBIAN_VERSION}-${TARGETARCH},target=/var/lib/apt/lists \
    # setup apt mirror follow deb-822 format
    sed -i.bak "s/deb.debian.org/${DEBIAN_APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources && \
    # debian的docker会自动删除下载的包 导致缓存失效 因此修改一下
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libvips
