//go:build with_audio

package cpullmapi

import "context"

type ASRResult struct {
	Text    string `json:"text"`
	Lang    string `json:"lang"`
	Emotion string `json:"emotion"`
}

type OfflineASRInferencer interface {
	Inferencer
	Transcribe(context.Context, []float32, string) (result ASRResult, err error)
	SampleRate() int
}

// ASRStreamResult carries the full transcript so far, not a delta: a
// transformer-backed ASR re-decodes its whole audio context each step, so text
// from an earlier step is regularly rewritten or withdrawn. Consumers replace
// their buffer with Text instead of appending to it.
type ASRStreamResult struct {
	Text    string   `json:"text"`
	Lang    string   `json:"lang,omitempty"`
	Counter uint64   `json:"counter,omitempty"`
	Start   *float64 `json:"t0,omitempty"`
	End     *float64 `json:"t1,omitempty"`
	Final   bool     `json:"final,omitempty"`
}

// ASRStreamConfig is the per-stream override set. The zero value defers to the
// model's factoryConfig.
type ASRStreamConfig struct {
	Hotwords    string `json:"hotwords"`
	RTrimTokens *uint  `json:"rTrimTokens,omitempty"`
}

// ASRStream is a single recognition session, owned by one goroutine. Feed and
// Flush block until the backend has finished with the audio, so a stream holds
// its caller for the whole session.
type ASRStream interface {
	Feed(samples []float32) (ASRStreamResult, error)
	Flush() (ASRStreamResult, error)
	Reset() error
	Close()
}

// StreamingASRInferencer transcribes audio incrementally. It is separate from
// OfflineASRInferencer because it exposes session state that offline
// transcription has no use for; an inferencer may implement both.
type StreamingASRInferencer interface {
	Inferencer

	SampleRate() int
	NewASRStream(ctx context.Context, config ASRStreamConfig) (ASRStream, error)
}
