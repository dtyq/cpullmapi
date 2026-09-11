//go:build with_audio

package testutil

import (
	"os"

	core "github.com/dtyq/cpullmapi"
	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/wav"
)

// ReadWavAsFloat32 decodes path, down mixes it to mono and resamples it to
// sampleRate.
func ReadWavAsFloat32(path string, sampleRate int) ([]float32, error) {
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
		streamer = core.NewBeepDownMixer(streamer, core.DownMixMethodAverage)
	}

	if format.SampleRate != beep.SampleRate(sampleRate) {
		streamer = beep.Resample(4, format.SampleRate, beep.SampleRate(sampleRate), streamer)
	}

	return core.BeepConvertSamplesToFloat32Array(streamer), nil
}
