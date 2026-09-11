//go:build with_audio

package crispasr

import (
	"context"
	"fmt"
	"testing"
	"time"

	core "github.com/dtyq/cpullmapi"
	"github.com/dtyq/cpullmapi/internal/testutil"
	"github.com/stretchr/testify/assert"
)

const (
	testInputWav   = "../../play/guangdonghua.wav"
	testModelPath  = "../../models/cstr/qwen3-asr-0.6b-GGUF/qwen3-asr-0.6b-q8_0.gguf"
	testSampleRate = 16000
)

func TestCrispASROfflineASRInferencer(t *testing.T) {
	testutil.RequireFiles(t, testInputWav, testModelPath)

	var err error

	var inferencer core.OfflineASRInferencer
	inferencer, err = NewCrispASROfflineASRInferencer(
		OfflineASRConfig{
			ModelPath:        testModelPath,
			ThreadNum:        4,
			MaxNewTokens:     4096,
			FrequencyPenalty: 0.8,
			SampleRate:       testSampleRate,
		},
	)
	if err != nil {
		t.Fatalf("failed to create onnx inferencer: %v", err)
	}
	defer inferencer.Close()

	pcm, err := testutil.ReadWavAsFloat32(testInputWav, testSampleRate)
	if err != nil {
		t.Fatalf("failed to read audio file: %v", err)
	}

	durationUs := time.Microsecond * time.Duration(len(pcm)*1000000/16000)
	startTime := time.Now()
	results, err := inferencer.Transcribe(context.Background(), pcm, "")
	endTime := time.Now()
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	fmt.Println(results)
	fmt.Printf("Audio duration: %v\n", durationUs)
	fmt.Printf("Transcription time: %v\n", endTime.Sub(startTime))
	fmt.Printf("Speed: %0.2fx\n", durationUs.Seconds()/endTime.Sub(startTime).Seconds())
}
