package crispasr

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	core "github.com/dtyq/cpullmapi"
	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/wav"
	"github.com/stretchr/testify/assert"
)

const testInputWav = "../../../guangdonghua.wav"

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

func TestCrispASROfflineASRInferencer(t *testing.T) {
	var err error

	var inferencer core.OfflineASRInferencer
	inferencer, err = NewCrispASROfflineASRInferencer(
		OfflineASRConfig{
			ModelPath:        "../../models/qwen3-asr-0.6b-q8_0.gguf",
			ThreadNum:        4,
			MaxNewTokens:     4096,
			FrequencyPenalty: 0.8,
			SampleRate:       16000,
		},
	)
	if err != nil {
		t.Fatalf("failed to create onnx inferencer: %v", err)
	}
	defer inferencer.Close()

	pcm, err := beepReadWav(testInputWav)
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
