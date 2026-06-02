package llamacppcgo

import (
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/wav"
	"github.com/stretchr/testify/assert"

	core "github.com/dtyq/cpullmapi"
)

// const testInputWav = "../../putonghua.wav"
// const testInputWav = "../../guangdonghua.wav"
// const testInputWav = "../../被讨厌的勇气.wav"
// const testInputWav = "../../Stable Diffusion.wav"
const testInputWav = "../../Rijndael.wav"

func beepReadWav(path string) ([]float32, error) {
	audioDataFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer audioDataFile.Close()

	stream, format, err := wav.Decode(audioDataFile)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var streamer beep.Streamer = stream

	if format.NumChannels > 1 {
		// do down mix
		streamer = core.NewBeepDownMixer(
			streamer,
			core.DownMixMethodAverage,
		)
	}

	// do SRC if needed
	if format.SampleRate != 16000 {
		streamer = beep.Resample(4, format.SampleRate, beep.SampleRate(16000), streamer)
	}

	float32Samples := core.BeepConvertSamplesToFloat32Array(streamer)

	return float32Samples, nil
}

func TestQwen3ASR(t *testing.T) {
	var err error
	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	ngl := int32(64)
	nctx := uint32(4096)
	lccSess, err := NewSession(SessionConfig{
		ModelPath:    modelPath,
		MMProjPath:   mmprojPath,
		NumGPULayers: &ngl,
		NCtx:         &nctx,
		Sampler: SamplerConfig{
			Kind: SamplerKindGreedy,
		},
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}
	defer lccSess.Close()

	sess := NewQwen3ASRSession(lccSess)

	err = sess.SetSystemPrompt("Rijndael Joan Daemen Vincent Rijmen")
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	const chunkSize = 15840 // 0.99s for 16kHz for Qwen3-ASR to avoid padding issues
	samples, err := beepReadWav(testInputWav)
	if err != nil {
		t.FailNow()
	}

	for sampleChunk := range slices.Chunk(samples, chunkSize) {
		lang, text, err := sess.FeedAudioSamples(sampleChunk, 4)
		if !assert.NoError(t, err) {
			t.FailNow()
		}

		fmt.Printf("Language: %s, Text: %s\n", lang, text)
	}
}
