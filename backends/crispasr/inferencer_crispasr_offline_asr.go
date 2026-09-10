package crispasr

import (
	"context"
	"fmt"

	crispasr "github.com/CrispStrobe/CrispASR/bindings/go"

	core "github.com/dtyq/cpullmapi"
)

type CrispASROfflineASRInferencer struct {
	session    *crispasr.CrispasrSession
	sampleRate int
}

// static assert the inferencer
var _ core.OfflineASRInferencer = (*CrispASROfflineASRInferencer)(nil)

type OfflineASRConfig struct {
	ModelPath        string  `json:"modelPath" yaml:"modelPath"`
	ThreadNum        uint    `json:"threadNum" yaml:"threadNum"`
	MaxNewTokens     uint    `json:"maxNewTokens" yaml:"maxNewTokens"`
	FrequencyPenalty float32 `json:"frequencyPenalty" yaml:"frequencyPenalty"`
	SampleRate       uint    `json:"sampleRate" yaml:"sampleRate"`
}

func NewCrispASROfflineASRInferencer(
	modelConfig OfflineASRConfig,
) (*CrispASROfflineASRInferencer, error) {
	sess, err := crispasr.SessionOpen(
		modelConfig.ModelPath,
		int(modelConfig.ThreadNum),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	if modelConfig.MaxNewTokens != 0 {
		err = sess.SetMaxNewTokens(int(modelConfig.MaxNewTokens))
		if err != nil {
			return nil, fmt.Errorf("failed to set max new tokens: %v", err)
		}
	}
	if modelConfig.FrequencyPenalty != 0 {
		err = sess.SetFrequencyPenalty(modelConfig.FrequencyPenalty)
		if err != nil {
			return nil, fmt.Errorf("failed to set frequency penalty: %v", err)
		}
	}

	return &CrispASROfflineASRInferencer{
		session:    sess,
		sampleRate: int(modelConfig.SampleRate),
	}, nil
}

func (i *CrispASROfflineASRInferencer) Close() {
	i.session.Close()
	i.session = nil
}

func (i *CrispASROfflineASRInferencer) GetCapabilities() []core.Capability {
	return []core.Capability{core.CapabilityOfflineASR}
}

func (i *CrispASROfflineASRInferencer) Transcribe(
	ctx context.Context,
	samples []float32,
	hotwords string,
) (result core.ASRResult, err error) {
	if i.session == nil {
		return core.ASRResult{}, fmt.Errorf("session is closed")
	}

	results, err := i.session.Transcribe(samples)
	if err != nil {
		return core.ASRResult{}, fmt.Errorf("failed to transcribe: %v", err)
	}
	if len(results.Segments) == 0 {
		return core.ASRResult{}, fmt.Errorf("no segments in transcription result")
	}

	return core.ASRResult{
		Text: results.Segments[0].Text,
	}, nil
}

func (i *CrispASROfflineASRInferencer) SampleRate() int {
	if i.sampleRate == 0 {
		return 16000
	}
	return i.sampleRate
}

func init() {
	core.InferencerFactoryMap["CrispASROfflineASR"] = NewCrispASROfflineASRInferencer
}
