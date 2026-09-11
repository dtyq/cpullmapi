package cpullmapi

import (
	"context"
	"fmt"
)

type Capability string

const (
	CapabilityImageSegmentation Capability = "image-seg"
	CapabilityOfflineASR        Capability = "offline-asr"
	CapabilityStreamingASR      Capability = "streaming-asr"
)

type Inferencer interface {
	GetCapabilities() []Capability

	Close()
}

// pickInferencer 取一个模型并断言它实现了 T。取到的槽位由 ctx 把着，整个请求
// 期间不释放。要在派发到 executor 之前调用：进了 executor 就改不了响应头了。
func pickInferencer[T any](
	pool *ResourcePool[Inferencer], ctx context.Context, modelName, capability string,
) (T, error) {
	var target T

	inferencer, err := pool.GetObj(ctx, modelName)
	if err != nil {
		return target, fmt.Errorf("model %s not found", modelName)
	}

	typed, ok := inferencer.(T)
	if !ok {
		return target, fmt.Errorf("model %s does not support %s", modelName, capability)
	}
	return typed, nil
}

var InferencerFactoryMap = map[string]any /*func(config [T any]) (Inferencer, error)*/ {}

type DummyInferencer struct {
}

func (i *DummyInferencer) GetCapabilities() []Capability {
	return []Capability{}
}

func (i *DummyInferencer) Close() {
}
