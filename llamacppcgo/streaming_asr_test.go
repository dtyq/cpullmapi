//go:build with_audio

package llamacppcgo

import (
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/dtyq/cpullmapi/internal/testutil"
)

// const testInputWav = "../play/putonghua.wav"
// const testInputWav = "../play/guangdonghua.wav"
// const testInputWav = "../play/被讨厌的勇气.wav"
// const testInputWav = "../play/Stable Diffusion.wav"
const testInputWav = "../play/Rijndael.wav"

func TestQwen3ASR(t *testing.T) {
	testutil.RequireFiles(t, testInputWav, libllamaPath, libmtmdPath, modelPath, mmprojPath)

	var err error
	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	SetLlamaLogFunc(func(level GGMLLogLevel, text string) {
		if level < GGML_LOG_LEVEL_WARN {
			return
		}
		os.Stderr.WriteString(text)
		os.Stderr.Sync()
	})

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
	samples, err := testutil.ReadWavAsFloat32(testInputWav, 16000)
	if err != nil {
		t.FailNow()
	}

	var lang, text string
	for sampleChunk := range slices.Chunk(samples, chunkSize) {
		lang, text, err = sess.FeedAudioSamples(sampleChunk, 4)
		if !assert.NoError(t, err) {
			t.FailNow()
		}

		fmt.Printf("Language: %s, Text: %s\n", lang, text)
	}
}
