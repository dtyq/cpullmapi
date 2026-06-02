package llamacppcgo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TODO: use enviroment variables for library paths in tests
const modelPath = "../models/ggml-org/Qwen3-ASR-0.6B-GGUF/Qwen3-ASR-0.6B-Q8_0.gguf"
const mmprojPath = "../models/ggml-org/Qwen3-ASR-0.6B-GGUF/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

func TestSessionCreate(t *testing.T) {
	var err error
	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	ngl := int32(-1)
	sess, err := NewSession(SessionConfig{
		ModelPath:    modelPath,
		MMProjPath:   mmprojPath,
		NumGPULayers: &ngl,
		Sampler: SamplerConfig{
			Kind:        SamplerKindDist,
			TopK:        5,
			TopP:        0.9,
			Temperature: 0.8,
			Seed:        12345,
		},
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	defer sess.Close()
}
