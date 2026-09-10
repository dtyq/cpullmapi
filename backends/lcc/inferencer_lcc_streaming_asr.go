//go:build with_audio

package lcc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	core "github.com/dtyq/cpullmapi"
	"github.com/dtyq/cpullmapi/llamacppcgo"
)

const defaultSampleRate = 16000

type StreamingASRConfig struct {
	LibllamaPath  string                    `json:"libllamaPath" yaml:"libllamaPath"`
	LibmtmdPath   string                    `json:"libmtmdPath" yaml:"libmtmdPath"`
	SessionConfig llamacppcgo.SessionConfig `json:"sessionConfig" yaml:"sessionConfig"`
	SystemPrompt  string                    `json:"systemPrompt" yaml:"systemPrompt"`
	SampleRate    uint                      `json:"sampleRate" yaml:"sampleRate"`
	RTrimTokens   uint                      `json:"rTrimTokens" yaml:"rTrimTokens"`
}

type LCCStreamingASRInferencer struct {
	session      *llamacppcgo.Session
	asrSession   *llamacppcgo.Qwen3ASRSession
	systemPrompt string
	sampleRate   int
	rTrimTokens  uint

	mu         sync.Mutex
	streamOpen bool
}

// static assert the inferencer
var _ core.StreamingASRInferencer = (*LCCStreamingASRInferencer)(nil)

func NewLCCStreamingASRInferencer(config StreamingASRConfig) (*LCCStreamingASRInferencer, error) {
	if err := llamacppcgo.LoadLibrary(config.LibllamaPath, config.LibmtmdPath); err != nil {
		return nil, fmt.Errorf("failed to load llama.cpp library: %w", err)
	}

	session, err := llamacppcgo.NewSession(config.SessionConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create llama.cpp session: %w", err)
	}

	sampleRate := int(config.SampleRate)
	if sampleRate == 0 {
		sampleRate = defaultSampleRate
	}

	return &LCCStreamingASRInferencer{
		session:      session,
		asrSession:   llamacppcgo.NewQwen3ASRSession(session),
		systemPrompt: config.SystemPrompt,
		sampleRate:   sampleRate,
		rTrimTokens:  config.RTrimTokens,
	}, nil
}

func (i *LCCStreamingASRInferencer) GetCapabilities() []core.Capability {
	return []core.Capability{core.CapabilityStreamingASR}
}

func (i *LCCStreamingASRInferencer) SampleRate() int {
	return i.sampleRate
}

func (i *LCCStreamingASRInferencer) Close() {
	i.mu.Lock()
	defer i.mu.Unlock()

	if i.asrSession == nil {
		return
	}
	i.asrSession.Close()
	i.asrSession = nil
	i.session = nil
}

func (i *LCCStreamingASRInferencer) NewASRStream(ctx context.Context, config core.ASRStreamConfig) (core.ASRStream, error) {
	i.mu.Lock()
	defer i.mu.Unlock()

	if i.asrSession == nil {
		return nil, errors.New("inferencer is closed")
	}
	if i.streamOpen {
		return nil, errors.New("inferencer already has an open stream")
	}

	prompt := strings.TrimSpace(i.systemPrompt + " " + config.Hotwords)
	if err := i.asrSession.SetSystemPrompt(prompt); err != nil {
		return nil, fmt.Errorf("failed to set system prompt: %w", err)
	}

	rTrimTokens := i.rTrimTokens
	if config.RTrimTokens != nil {
		rTrimTokens = *config.RTrimTokens
	}

	i.streamOpen = true
	return &lccASRStream{
		inferencer:  i,
		asrSession:  i.asrSession,
		ctx:         ctx,
		prompt:      prompt,
		rTrimTokens: rTrimTokens,
	}, nil
}

func (i *LCCStreamingASRInferencer) closeStream() {
	i.mu.Lock()
	defer i.mu.Unlock()

	i.streamOpen = false
}

type lccASRStream struct {
	inferencer  *LCCStreamingASRInferencer
	asrSession  *llamacppcgo.Qwen3ASRSession
	ctx         context.Context
	prompt      string
	rTrimTokens uint

	counter  uint64
	lastText string
}

func (s *lccASRStream) Feed(samples []float32) (core.ASRStreamResult, error) {
	if err := s.ctx.Err(); err != nil {
		return core.ASRStreamResult{}, err
	}

	lang, text, err := s.asrSession.FeedAudioSamples(samples, s.rTrimTokens)
	if err != nil {
		return core.ASRStreamResult{}, fmt.Errorf("failed to feed audio samples: %w", err)
	}

	return s.snapshot(lang, text), nil
}

func (s *lccASRStream) Flush() (core.ASRStreamResult, error) {
	if err := s.ctx.Err(); err != nil {
		return core.ASRStreamResult{}, err
	}

	lang, text, err := s.asrSession.Flush(s.rTrimTokens)
	if err != nil {
		return core.ASRStreamResult{}, fmt.Errorf("failed to flush stream: %w", err)
	}

	result := s.snapshot(lang, text)
	result.Final = true
	return result, nil
}

// Reset re-applies the prompt, which clears the KV memory and restarts the
// audio planner. Qwen3ASRSession.Reset would do the same but rejects an empty
// prompt, and an empty prompt is legal here.
func (s *lccASRStream) Reset() error {
	if err := s.asrSession.SetSystemPrompt(s.prompt); err != nil {
		return fmt.Errorf("failed to reset stream: %w", err)
	}

	s.counter = 0
	s.lastText = ""
	return nil
}

// Close releases the stream but keeps the llama.cpp session alive: the
// Qwen3ASRSession owns it, and the next stream reuses it via SetSystemPrompt.
func (s *lccASRStream) Close() {
	s.inferencer.closeStream()
}

func (s *lccASRStream) snapshot(lang string, text string) core.ASRStreamResult {
	if text != s.lastText {
		s.counter++
		s.lastText = text
	}

	return core.ASRStreamResult{
		Text:    text,
		Lang:    lang,
		Counter: s.counter,
	}
}

func init() {
	core.InferencerFactoryMap["LCCStreamingASR"] = NewLCCStreamingASRInferencer
}
