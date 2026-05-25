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
