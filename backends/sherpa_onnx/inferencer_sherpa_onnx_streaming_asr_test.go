//go:build with_audio

package sherpa_onnx

import (
	"context"
	"fmt"
	"slices"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"github.com/stretchr/testify/assert"

	core "github.com/dtyq/cpullmapi"
	"github.com/dtyq/cpullmapi/internal/testutil"
)

const (
	sherpaStreamingModelDir = "../../models/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en"
	sherpaStreamingWav      = "../../play/guangdonghua.wav"

	// 200ms at 16kHz.
	sherpaStreamingChunk = 3200
)

func newSherpaStreamingInferencer(t *testing.T) *SherpaONNXStreamingASRInferencer {
	t.Helper()

	inferencer, err := NewSherpaONNXStreamingASRInferencer(SherpaONNXStreamingASRConfig{
		RecognizerConfig: sherpa.OnlineRecognizerConfig{
			FeatConfig: sherpa.FeatureConfig{
				SampleRate: defaultSherpaSampleRate,
				FeatureDim: 80,
			},
			ModelConfig: sherpa.OnlineModelConfig{
				Paraformer: sherpa.OnlineParaformerModelConfig{
					Encoder: sherpaStreamingModelDir + "/encoder.int8.onnx",
					Decoder: sherpaStreamingModelDir + "/decoder.int8.onnx",
				},
				Tokens:     sherpaStreamingModelDir + "/tokens.txt",
				NumThreads: 2,
				Provider:   "cpu",
			},
			DecodingMethod:          "greedy_search",
			EnableEndpoint:          1,
			Rule1MinTrailingSilence: 2.4,
			Rule2MinTrailingSilence: 1.2,
			Rule3MinUtteranceLength: 20,
		},
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	return inferencer
}

func sherpaFeedAll(t *testing.T, stream core.ASRStream, samples []float32) {
	t.Helper()

	var last core.ASRStreamResult
	for chunk := range slices.Chunk(samples, sherpaStreamingChunk) {
		result, err := stream.Feed(chunk)
		if !assert.NoError(t, err) {
			t.FailNow()
		}
		if result.Text != last.Text {
			fmt.Printf("counter=%d t0=%v t1=%v eou=%v text=%q\n",
				result.Counter, result.Start, result.End, result.EndOfUtterance, result.Text)
		}
		last = result
	}
}

func TestSherpaONNXStreamingASRInferencer(t *testing.T) {
	testutil.RequireFiles(t, sherpaStreamingModelDir, sherpaStreamingWav)

	samples, err := testutil.ReadWavAsFloat32(sherpaStreamingWav, defaultSherpaSampleRate)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	inferencer := newSherpaStreamingInferencer(t)
	defer inferencer.Close()

	if !assert.Equal(t, []core.Capability{core.CapabilityStreamingASR}, inferencer.GetCapabilities()) {
		t.FailNow()
	}
	assert.Equal(t, defaultSherpaSampleRate, inferencer.SampleRate())

	stream, err := inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{})
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	_, err = inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{})
	assert.ErrorContains(t, err, "already has an open stream")

	sherpaFeedAll(t, stream, samples)

	final, err := stream.Flush()
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	fmt.Printf("final counter=%d t0=%v t1=%v text=%q\n", final.Counter, final.Start, final.End, final.Text)

	assert.True(t, final.Final)
	assert.NotEmpty(t, final.Text)

	// This streaming paraformer holds a whole chunk in flight and emits no
	// trailing silence of its own, so the final phoneme of the utterance is
	// missing from the result. Feeding silence to flush it was measured both
	// ways: below ~8s the output does not change at all, and at ~8s the model
	// recovers that phoneme but then hallucinates repetition across the
	// silence, which is worse. There is no setting that yields the clean tail,
	// so the backend does not pretend otherwise and this asserts what is
	// actually stable instead.
	assert.Contains(t, final.Text, "帮我查下")

	_, err = stream.Feed(samples[:sherpaStreamingChunk])
	assert.ErrorContains(t, err, "no more audio")

	stream.Close()

	// The recognizer outlives the stream, so the same inferencer can serve
	// another utterance.
	reused, err := inferencer.NewASRStream(context.Background(), core.ASRStreamConfig{})
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	defer reused.Close()

	sherpaFeedAll(t, reused, samples)

	reusedFinal, err := reused.Flush()
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	fmt.Printf("reused final counter=%d text=%q\n", reusedFinal.Counter, reusedFinal.Text)
	assert.Equal(t, final.Text, reusedFinal.Text)
}
