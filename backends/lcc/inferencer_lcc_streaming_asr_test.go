//go:build with_audio

package lcc

import (
	"context"
	"fmt"
	"slices"
	"testing"

	core "github.com/dtyq/cpullmapi"
	"github.com/dtyq/cpullmapi/internal/testutil"
	"github.com/dtyq/cpullmapi/llamacppcgo"
	"github.com/stretchr/testify/assert"
)

const (
	libllamaPath = "../../third_party/llama.cpp/build_go/bin/libllama.so"
	libmtmdPath  = "../../third_party/llama.cpp/build_go/bin/libmtmd.so"

	modelPath  = "../../models/ggml-org/Qwen3-ASR-0.6B-GGUF/Qwen3-ASR-0.6B-Q8_0.gguf"
	mmprojPath = "../../models/ggml-org/Qwen3-ASR-0.6B-GGUF/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

	testInputWav = "../../play/guangdonghua.wav"

	testChunkSize = 15840
)

func newTestInferencer(t *testing.T) *LCCStreamingASRInferencer {
	t.Helper()

	ngl := int32(64)
	nctx := uint32(4096)

	inferencer, err := NewLCCStreamingASRInferencer(StreamingASRConfig{
		LibllamaPath: libllamaPath,
		LibmtmdPath:  libmtmdPath,
		SampleRate:   defaultSampleRate,
		RTrimTokens:  4,
		SessionConfig: llamacppcgo.SessionConfig{
			ModelPath:    modelPath,
			MMProjPath:   mmprojPath,
			NumGPULayers: &ngl,
			NCtx:         &nctx,
			Sampler: llamacppcgo.SamplerConfig{
				Kind: llamacppcgo.SamplerKindGreedy,
			},
		},
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	return inferencer
}

func feedAll(t *testing.T, stream core.ASRStream, samples []float32) core.ASRStreamResult {
	t.Helper()

	var last core.ASRStreamResult
	for chunk := range slices.Chunk(samples, testChunkSize) {
		result, err := stream.Feed(chunk)
		if !assert.NoError(t, err) {
			t.FailNow()
		}
		if result.Text != last.Text {
			fmt.Printf("counter=%d lang=%q text=%q\n", result.Counter, result.Lang, result.Text)
		}
		last = result
	}

	return last
}

func TestLCCStreamingASRInferencer(t *testing.T) {
	testutil.RequireFiles(t, libllamaPath, libmtmdPath, modelPath, mmprojPath, testInputWav)

	samples, err := testutil.ReadWavAsFloat32(testInputWav, defaultSampleRate)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	inferencer := newTestInferencer(t)
	defer inferencer.Close()

	if !assert.Equal(t, []core.Capability{core.CapabilityStreamingASR}, inferencer.GetCapabilities()) {
		t.FailNow()
	}

	stream, err := inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{
		Hotwords: "广东话",
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	assert.Equal(t, defaultSampleRate, inferencer.SampleRate())

	_, err = inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{})
	assert.ErrorContains(t, err, "already has an open stream")

	feedAll(t, stream, samples)

	final, err := stream.Flush()
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	fmt.Printf("final counter=%d lang=%q text=%q\n", final.Counter, final.Lang, final.Text)
	assert.True(t, final.Final)
	assert.NotEmpty(t, final.Text)
	assert.GreaterOrEqual(t, final.Counter, uint64(1))

	stream.Close()

	// The underlying llama.cpp session survives a stream, so the same
	// inferencer can serve another utterance.
	reused, err := inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{})
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	defer reused.Close()

	if !assert.NoError(t, reused.Reset()) {
		t.FailNow()
	}

	feedAll(t, reused, samples[:testChunkSize])

	reusedFinal, err := reused.Flush()
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	fmt.Printf("reused final counter=%d lang=%q text=%q\n", reusedFinal.Counter, reusedFinal.Lang, reusedFinal.Text)
	assert.True(t, reusedFinal.Final)
}
