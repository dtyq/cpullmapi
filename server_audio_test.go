//go:build with_audio

package cpullmapi

import (
	"testing"
	"time"
)

const (
	testStreamModel      = "StreamingASR"
	testStreamSampleRate = 16000
)

func newAudioTestServer(t *testing.T, inferencers map[string]Inferencer, timeout time.Duration) *testServer {
	t.Helper()

	ts := newTestServer(t, inferencers, func(config *Config) {
		config.Misc.StreamingASRTimeout = timeout
	})
	ts.api.POST("/transcribe", ts.server.transcribeHandler)
	ts.api.POST("/transcribe/stream", ts.server.transcribeStreamHandler)
	ts.api.GET("/transcribe/realtime", ts.server.transcribeRealtimeHandler)
	return ts
}
