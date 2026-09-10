package sherpa_onnx

import (
	"context"
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	core "github.com/dtyq/cpullmapi"
)

type SherpaONNXOfflineASRInferencer struct {
	recognizer *sherpa.OfflineRecognizer
	sampleRate int
}

// static assert the inferencer
var _ core.OfflineASRInferencer = (*SherpaONNXOfflineASRInferencer)(nil)

func NewSherpaONNXOfflineASRInferencer(
	modelConfig sherpa.OfflineRecognizerConfig,
) (*SherpaONNXOfflineASRInferencer, error) {
	recognizer := sherpa.NewOfflineRecognizer(&modelConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create recognizer")
	}

	return &SherpaONNXOfflineASRInferencer{
		sampleRate: modelConfig.FeatConfig.SampleRate,
		recognizer: recognizer,
	}, nil
}

func (i *SherpaONNXOfflineASRInferencer) Close() {
	sherpa.DeleteOfflineRecognizer(i.recognizer)
}

func (i *SherpaONNXOfflineASRInferencer) GetCapabilities() []core.Capability {
	return []core.Capability{core.CapabilityOfflineASR}
}

func (i *SherpaONNXOfflineASRInferencer) Transcribe(
	ctx context.Context,
	samples []float32,
	hotwords string,
) (result core.ASRResult, err error) {
	stream := sherpa.NewOfflineStream(i.recognizer)
	if stream == nil {
		return core.ASRResult{}, fmt.Errorf("failed to create stream")
	}
	defer sherpa.DeleteOfflineStream(stream)

	if hotwords != "" && stream.HasOption("hotwords") {
		stream.SetOption("hotwords", hotwords)
	}

	stream.AcceptWaveform(i.sampleRate, samples)
	i.recognizer.Decode(stream)
	ret := stream.GetResult()
	return core.ASRResult{
		Text:    ret.Text,
		Lang:    ret.Lang,
		Emotion: ret.Emotion,
	}, nil
}

func (i *SherpaONNXOfflineASRInferencer) SampleRate() int {
	return i.sampleRate
}

func init() {
	core.InferencerFactoryMap["SherpaONNXOfflineASR"] = NewSherpaONNXOfflineASRInferencer
}
