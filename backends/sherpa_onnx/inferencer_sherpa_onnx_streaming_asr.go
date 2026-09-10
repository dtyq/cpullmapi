//go:build with_audio

package sherpa_onnx

import (
	"context"
	"errors"
	"fmt"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	core "github.com/dtyq/cpullmapi"
)

const defaultSherpaSampleRate = 16000

// SherpaONNXStreamingASRConfig configures the streaming recognizer.
type SherpaONNXStreamingASRConfig struct {
	RecognizerConfig sherpa.OnlineRecognizerConfig `json:"recognizerConfig" yaml:"recognizerConfig"`
}

type SherpaONNXStreamingASRInferencer struct {
	recognizer *sherpa.OnlineRecognizer
	sampleRate int

	mu         sync.Mutex
	streamOpen bool
}

// static assert the inferencer
var _ core.StreamingASRInferencer = (*SherpaONNXStreamingASRInferencer)(nil)

func NewSherpaONNXStreamingASRInferencer(
	config SherpaONNXStreamingASRConfig,
) (*SherpaONNXStreamingASRInferencer, error) {
	recognizer := sherpa.NewOnlineRecognizer(&config.RecognizerConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create recognizer")
	}

	sampleRate := config.RecognizerConfig.FeatConfig.SampleRate
	if sampleRate == 0 {
		sampleRate = defaultSherpaSampleRate
	}

	return &SherpaONNXStreamingASRInferencer{
		recognizer: recognizer,
		sampleRate: sampleRate,
	}, nil
}

func (i *SherpaONNXStreamingASRInferencer) GetCapabilities() []core.Capability {
	return []core.Capability{core.CapabilityStreamingASR}
}

func (i *SherpaONNXStreamingASRInferencer) SampleRate() int {
	return i.sampleRate
}

func (i *SherpaONNXStreamingASRInferencer) Close() {
	i.mu.Lock()
	defer i.mu.Unlock()

	if i.recognizer == nil {
		return
	}
	sherpa.DeleteOnlineRecognizer(i.recognizer)
	i.recognizer = nil
}

func (i *SherpaONNXStreamingASRInferencer) NewASRStream(ctx context.Context, config core.ASRStreamConfig) (core.ASRStream, error) {
	i.mu.Lock()
	defer i.mu.Unlock()

	if i.recognizer == nil {
		return nil, errors.New("inferencer is closed")
	}
	if i.streamOpen {
		return nil, errors.New("inferencer already has an open stream")
	}

	stream := sherpa.NewOnlineStream(i.recognizer)
	if stream == nil {
		return nil, errors.New("failed to create stream")
	}

	s := &sherpaONNXASRStream{
		inferencer: i,
		recognizer: i.recognizer,
		stream:     stream,
		sampleRate: i.sampleRate,
		ctx:        ctx,
		hotwords:   config.Hotwords,
	}
	s.applyHotwords()

	i.streamOpen = true
	return s, nil
}

func (i *SherpaONNXStreamingASRInferencer) closeStream() {
	i.mu.Lock()
	defer i.mu.Unlock()

	i.streamOpen = false
}

type sherpaONNXASRStream struct {
	inferencer *SherpaONNXStreamingASRInferencer
	recognizer *sherpa.OnlineRecognizer
	stream     *sherpa.OnlineStream
	sampleRate int
	ctx        context.Context
	hotwords   string

	flushed  bool
	counter  uint64
	lastText string
}

func (s *sherpaONNXASRStream) Feed(samples []float32) (core.ASRStreamResult, error) {
	if err := s.ctx.Err(); err != nil {
		return core.ASRStreamResult{}, err
	}
	if s.flushed {
		return core.ASRStreamResult{}, errors.New("stream is flushed and accepts no more audio")
	}
	if len(samples) == 0 {
		return s.snapshot(s.recognizer.GetResult(s.stream)), nil
	}

	s.stream.AcceptWaveform(s.sampleRate, samples)
	s.decode()

	result := s.snapshot(s.recognizer.GetResult(s.stream))

	if s.recognizer.IsEndpoint(s.stream) {
		s.recognizer.Reset(s.stream)
		if result.Text != "" {
			result.EndOfUtterance = true
		}
	}

	return result, nil
}

func (s *sherpaONNXASRStream) Flush() (core.ASRStreamResult, error) {
	if err := s.ctx.Err(); err != nil {
		return core.ASRStreamResult{}, err
	}

	// Set unconditionally: HasOption reports whether the option is already set,
	// not whether the model supports it, so guarding on it never fired.
	s.stream.SetOption("is_final", "1")
	s.stream.InputFinished()
	s.decode()
	s.flushed = true

	result := s.snapshot(s.recognizer.GetResult(s.stream))
	result.Final = true
	return result, nil
}

// Reset builds a fresh stream: Flush calls InputFinished, which permanently
// closes the previous one.
func (s *sherpaONNXASRStream) Reset() error {
	stream := sherpa.NewOnlineStream(s.recognizer)
	if stream == nil {
		return errors.New("failed to create stream")
	}

	if s.stream != nil {
		sherpa.DeleteOnlineStream(s.stream)
	}
	s.stream = stream
	s.flushed = false
	s.counter = 0
	s.lastText = ""
	s.applyHotwords()

	return nil
}

func (s *sherpaONNXASRStream) Close() {
	if s.stream != nil {
		sherpa.DeleteOnlineStream(s.stream)
		s.stream = nil
	}
	s.inferencer.closeStream()
}

func (s *sherpaONNXASRStream) decode() {
	for s.recognizer.IsReady(s.stream) {
		s.recognizer.Decode(s.stream)
	}
}

func (s *sherpaONNXASRStream) applyHotwords() {
	if s.hotwords != "" && s.stream.HasOption("hotwords") {
		s.stream.SetOption("hotwords", s.hotwords)
	}
}

func (s *sherpaONNXASRStream) snapshot(result *sherpa.OnlineRecognizerResult) core.ASRStreamResult {
	var text string
	var start, end *float64

	if result != nil {
		text = result.Text
		if n := len(result.Timestamps); n > 0 {
			first, last := float64(result.Timestamps[0]), float64(result.Timestamps[n-1])
			start, end = &first, &last
		}
	}

	if text != s.lastText {
		s.counter++
		s.lastText = text
	}

	return core.ASRStreamResult{
		Text:    text,
		Counter: s.counter,
		Start:   start,
		End:     end,
	}
}

func init() {
	core.InferencerFactoryMap["SherpaONNXStreamingASR"] = NewSherpaONNXStreamingASRInferencer
}
