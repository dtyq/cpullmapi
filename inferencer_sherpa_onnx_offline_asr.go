package cpullmapi

import (
	"context"
	"fmt"

	"github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

type SampleRater interface {
	SampleRate() int
}

type SherpaONNXOfflineASRInferencer struct {
	DummyInferencer

	recognizer *sherpa_onnx.OfflineRecognizer
	sampleRate int
}

func NewSherpaONNXOfflineASRInferencer(
	modelConfig sherpa_onnx.OfflineRecognizerConfig,
) (*SherpaONNXOfflineASRInferencer, error) {
	recognizer := sherpa_onnx.NewOfflineRecognizer(&modelConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create recognizer")
	}

	return &SherpaONNXOfflineASRInferencer{
		sampleRate: modelConfig.FeatConfig.SampleRate,
		recognizer: recognizer,
	}, nil
}

func (i *SherpaONNXOfflineASRInferencer) Close() {
	sherpa_onnx.DeleteOfflineRecognizer(i.recognizer)
}

func (i *SherpaONNXOfflineASRInferencer) GetCapabilities() []Capability {
	return []Capability{CapabilityOfflineASR}
}

func (i *SherpaONNXOfflineASRInferencer) Transcribe(
	ctx context.Context,
	samples []float32,
	hotwords string,
) (result ASRResult, err error) {
	stream := sherpa_onnx.NewOfflineStream(i.recognizer)
	if stream == nil {
		return ASRResult{}, fmt.Errorf("failed to create stream")
	}
	defer sherpa_onnx.DeleteOfflineStream(stream)

	if hotwords != "" && stream.HasOption("hotwords") {
		stream.SetOption("hotwords", hotwords)
	}

	stream.AcceptWaveform(i.sampleRate, samples)
	i.recognizer.Decode(stream)
	ret := stream.GetResult()
	return ASRResult{
		Text:    ret.Text,
		Lang:    ret.Lang,
		Emotion: ret.Emotion,
	}, nil
}

func (i *SherpaONNXOfflineASRInferencer) SampleRate() int {
	return i.sampleRate
}

func init() {
	InferencerFactoryMap["SherpaONNXOfflineASR"] = NewSherpaONNXOfflineASRInferencer
}
