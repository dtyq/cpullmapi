//go:build with_audio

package cpullmapi

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// sseEvents 解析 text/event-stream，返回每条 data: 后面的内容。
func sseEvents(t *testing.T, body io.Reader) []string {
	t.Helper()

	var events []string
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if payload, found := strings.CutPrefix(line, "data: "); found {
			events = append(events, payload)
		}
	}
	require.NoError(t, scanner.Err())
	return events
}

func TestTranscribeStreamRequiresToken(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel, nil)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusUnauthorized, recorder.Code)
}

func TestTranscribeStreamRejectsMissingModelName(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream", nil)
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "modelName is required", recorder.Header().Get("X-Error"))
}

func TestTranscribeStreamRejectsUnknownModel(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName=nope", nil)
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := serveRequest(ts, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model nope not found", recorder.Header().Get("X-Error"))
}

func TestTranscribeStreamRejectsNonStreamingModel(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		"OfflineASR": &fakeOfflineInferencer{},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName=OfflineASR", nil)
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := serveRequest(ts, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model OfflineASR does not support streaming asr", recorder.Header().Get("X-Error"))
}

func TestTranscribeStreamRejectsWrongSampleRate(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: 8000},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel, nil)
	req.Header.Set("Authorization", "Token "+testToken)
	req.Header.Set("Content-Type", "audio/L16;rate=16000")
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Contains(t, recorder.Header().Get("X-Error"), "wants 8000 Hz, got 16000 Hz")
}

func TestTranscribeStreamRejectsBadContentType(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel, nil)
	req.Header.Set("Authorization", "Token "+testToken)
	req.Header.Set("Content-Type", "audio/wav")
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Contains(t, recorder.Header().Get("X-Error"), "unsupported content type")
}

func TestTranscribeStreamPassesConfigToInferencer(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	req := httptest.NewRequest(http.MethodPost,
		"/api/v1/transcribe/stream?modelName="+testStreamModel+"&hotwords=a,b&rTrimTokens=7",
		strings.NewReader(string(pcmBytes(1600))))
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "a,b", inferencer.streamConfig().Hotwords)
	rTrim := inferencer.streamConfig().RTrimTokens
	require.NotNil(t, rTrim)
	assert.Equal(t, uint(7), *rTrim)
}

func TestTranscribeStreamRejectsBadRTrimTokens(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodPost,
		"/api/v1/transcribe/stream?modelName="+testStreamModel+"&rTrimTokens=abc", nil)
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Contains(t, recorder.Header().Get("X-Error"), "invalid rTrimTokens")
}

func TestTranscribeStreamEmitsSSEAndDone(t *testing.T) {
	inferencer := &fakeStreamingInferencer{
		sampleRate: testStreamSampleRate,
		stream: &fakeASRStream{results: []ASRStreamResult{
			{Text: "帮我", Counter: 1},
			{Text: "帮我查下", Counter: 2},
		}},
	}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	// 3200 个采样 = 200ms，按 100ms 一块应该 Feed 两次。
	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel,
		strings.NewReader(string(pcmBytes(3200))))
	req.Header.Set("Authorization", "Token "+testToken)
	req.Header.Set("Content-Type", "audio/L16;rate=16000;channels=1")
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "text/event-stream", recorder.Header().Get("Content-Type"))

	events := sseEvents(t, recorder.Body)
	require.Len(t, events, 4, "two transcript events, the final one, then [DONE]")
	assert.Equal(t, streamSSEDone, events[len(events)-1])

	var first ASRStreamResult
	require.NoError(t, json.Unmarshal([]byte(events[0]), &first))
	assert.Equal(t, "帮我", first.Text)
	assert.Equal(t, uint64(1), first.Counter)

	var last ASRStreamResult
	require.NoError(t, json.Unmarshal([]byte(events[len(events)-2]), &last))
	assert.Equal(t, "final", last.Text)
	assert.True(t, last.Final)

	assert.True(t, inferencer.stream.isFlushed())
	assert.True(t, inferencer.stream.isClosed(), "the stream must be closed when the request ends")
}

func TestTranscribeStreamSkipsUnchangedResults(t *testing.T) {
	inferencer := &fakeStreamingInferencer{
		sampleRate: testStreamSampleRate,
		stream: &fakeASRStream{results: []ASRStreamResult{
			{Text: "same", Counter: 1},
			{Text: "same", Counter: 2},
			{Text: "same", Counter: 3},
		}},
	}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel,
		strings.NewReader(string(pcmBytes(4800))))
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	// 三次文本相同，只应推一次；加上 final 和 [DONE] 共 3 条。
	events := sseEvents(t, recorder.Body)
	require.Len(t, events, 3)
	assert.Equal(t, streamSSEDone, events[2])
}

func TestTranscribeStreamStopsAfterTimeout(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, 50*time.Millisecond)

	// httptest.NewRecorder 不支持读超时，得走真的连接。
	httpServer := httptest.NewServer(ts.router)
	t.Cleanup(httpServer.Close)

	// 发一段音频后就磨蹭，让服务端的读超时先触发。
	body := &stallingBody{}
	req, err := http.NewRequest(http.MethodPost,
		httpServer.URL+"/api/v1/transcribe/stream?modelName="+testStreamModel, body)
	require.NoError(t, err)
	req.Header.Set("Authorization", "Token "+testToken)
	req.Header.Set("Content-Type", "audio/L16;rate=16000")

	done := make(chan string, 1)
	go func() {
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			done <- ""
			return
		}
		defer resp.Body.Close()

		data, _ := io.ReadAll(resp.Body)
		done <- string(data)
	}()

	select {
	case data := <-done:
		// 超时不是音频结束，所以不该有 final 和 [DONE]。
		assert.NotContains(t, data, streamSSEDone)
		assert.NotContains(t, data, `"final":true`)
	case <-time.After(10 * time.Second):
		t.Fatal("handler did not return after the peer stopped sending")
	}

	assert.False(t, inferencer.stream.isFlushed(), "a timeout must not look like end of audio")
}

// stallingBody 先给一段音频，之后就慢慢燉。
type stallingBody struct {
	sent bool
}

func (b *stallingBody) Read(p []byte) (int, error) {
	if !b.sent {
		b.sent = true
		return copy(p, pcmBytes(1600)), nil
	}

	// 服务端的读超时比这里短得多，会先把它踢掉。
	time.Sleep(500 * time.Millisecond)
	return 0, io.EOF
}

func TestTranscribeStreamReturns500WhenStreamCannotStart(t *testing.T) {
	inferencer := &fakeStreamingInferencer{
		sampleRate:   testStreamSampleRate,
		streamErrors: []error{assert.AnError},
	}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcribe/stream?modelName="+testStreamModel,
		strings.NewReader(string(pcmBytes(100))))
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.Equal(t, "failed to create stream", recorder.Header().Get("X-Error"))
}

// dialRealtime 起一个真 HTTP 服务，用 WebSocket 连上去。
func dialRealtime(t *testing.T, ts *testServer, opts *websocket.DialOptions) (*websocket.Conn, *httptest.Server) {
	t.Helper()

	httpServer := httptest.NewServer(ts.router)
	t.Cleanup(httpServer.Close)

	url := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/transcribe/realtime?modelName=" + testStreamModel

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(ctx, url, opts)
	require.NoError(t, err)
	t.Cleanup(func() { conn.CloseNow() })

	return conn, httpServer
}

func readTranscript(t *testing.T, ctx context.Context, conn *websocket.Conn) wsTranscriptMessage {
	t.Helper()

	typ, data, err := conn.Read(ctx)
	require.NoError(t, err)
	require.Equal(t, websocket.MessageText, typ)

	var message wsTranscriptMessage
	require.NoError(t, json.Unmarshal(data, &message))
	return message
}

func TestTranscribeRealtimeRejectsMissingModelName(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/transcribe/realtime", nil)
	req.Header.Set("Authorization", "Token "+testToken)
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "modelName is required", recorder.Header().Get("X-Error"))
}

func TestTranscribeRealtimeRejectsBadToken(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testStreamModel: &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, time.Second)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/transcribe/realtime?modelName="+testStreamModel, nil)
	req.Header.Set("Authorization", "Token wrong")
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusUnauthorized, recorder.Code)
}

func TestTranscribeRealtimeAcceptsSubprotocolToken(t *testing.T) {
	inferencer := &fakeStreamingInferencer{
		sampleRate: testStreamSampleRate,
		stream:     &fakeASRStream{results: []ASRStreamResult{{Text: "hi", Counter: 1}}},
	}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	// 不带 Authorization，只靠子协议里的 token。
	conn, _ := dialRealtime(t, ts, &websocket.DialOptions{
		Subprotocols: []string{"realtime", "openai-insecure-api-key." + testToken},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, conn.Write(ctx, websocket.MessageBinary, pcmBytes(1600)))
	require.NoError(t, conn.Write(ctx, websocket.MessageText, []byte(`{"type":"input_audio_buffer.commit"}`)))

	message := readTranscript(t, ctx, conn)
	assert.Equal(t, "transcript", message.Type)
	assert.Equal(t, "hi", message.Text)

	final := readTranscript(t, ctx, conn)
	assert.True(t, final.Final)
	assert.Equal(t, "final", final.Text)
}

func TestTranscribeRealtimeSessionUpdate(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	conn, _ := dialRealtime(t, ts, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": []string{"Token " + testToken}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, conn.Write(ctx, websocket.MessageText,
		[]byte(`{"type":"session.update","hotwords":"热词","rTrimTokens":3}`)))
	require.NoError(t, conn.Write(ctx, websocket.MessageBinary, pcmBytes(1600)))
	require.NoError(t, conn.Write(ctx, websocket.MessageText, []byte(`{"type":"input_audio_buffer.commit"}`)))

	readTranscript(t, ctx, conn)

	config := inferencer.streamConfig()
	assert.Equal(t, "热词", config.Hotwords)
	require.NotNil(t, config.RTrimTokens)
	assert.Equal(t, uint(3), *config.RTrimTokens)
}

func TestTranscribeRealtimeCommitWithoutSessionUpdate(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	conn, _ := dialRealtime(t, ts, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": []string{"Token " + testToken}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 第一帧直接是音频，没有 session.update。
	require.NoError(t, conn.Write(ctx, websocket.MessageBinary, pcmBytes(3200)))
	require.NoError(t, conn.Write(ctx, websocket.MessageText, []byte(`{"type":"input_audio_buffer.commit"}`)))

	readTranscript(t, ctx, conn)

	assert.Empty(t, inferencer.streamConfig().Hotwords)
	assert.Equal(t, 3200, len(inferencer.stream.fedSamples()), "audio sent before any control frame must not be dropped")
}

func TestTranscribeRealtimeRejectsUnknownControlFrame(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	conn, _ := dialRealtime(t, ts, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": []string{"Token " + testToken}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, conn.Write(ctx, websocket.MessageBinary, pcmBytes(1600)))
	require.NoError(t, conn.Write(ctx, websocket.MessageText, []byte(`{"type":"nonsense"}`)))

	_, _, err := conn.Read(ctx)
	require.Error(t, err)
	assert.Equal(t, websocket.StatusInternalError, websocket.CloseStatus(err))
}

func TestTranscribeRealtimeClosesOnPeerClose(t *testing.T) {
	inferencer := &fakeStreamingInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testStreamModel: inferencer}, time.Second)

	conn, _ := dialRealtime(t, ts, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": []string{"Token " + testToken}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, conn.Write(ctx, websocket.MessageBinary, pcmBytes(1600)))
	// 直接关掉，不发 commit：也算音频结束。
	require.NoError(t, conn.Close(websocket.StatusNormalClosure, ""))

	_, _, err := conn.Read(ctx)
	require.Error(t, err)

	assert.True(t, inferencer.stream.isFlushed(), "peer close should be treated as end of audio")
}
