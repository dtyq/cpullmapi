//go:build with_audio

package cpullmapi

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testOfflineModel = "OfflineASR"

// fakeOfflineInferencer 只实现离线识别，不实现流式，用来验证类型检查。
type fakeOfflineInferencer struct {
	sampleRate int

	mu       sync.Mutex
	hotwords []string
	samples  int
	err      error
	result   ASRResult
}

func (i *fakeOfflineInferencer) GetCapabilities() []Capability {
	return []Capability{CapabilityOfflineASR}
}

func (i *fakeOfflineInferencer) Close() {}

func (i *fakeOfflineInferencer) SampleRate() int {
	if i.sampleRate == 0 {
		return testStreamSampleRate
	}
	return i.sampleRate
}

func (i *fakeOfflineInferencer) Transcribe(
	_ context.Context, samples []float32, hotwords string,
) (ASRResult, error) {
	i.mu.Lock()
	defer i.mu.Unlock()

	i.samples = len(samples)
	i.hotwords = append(i.hotwords, hotwords)
	return i.result, i.err
}

func (i *fakeOfflineInferencer) call() (int, string) {
	i.mu.Lock()
	defer i.mu.Unlock()

	if len(i.hotwords) == 0 {
		return i.samples, ""
	}
	return i.samples, i.hotwords[len(i.hotwords)-1]
}

// wavBytes 拼一个单声道 16bit 的最小 WAVE 文件。
func wavBytes(sampleRate int, samples []int16) []byte {
	data := new(bytes.Buffer)
	for _, sample := range samples {
		_ = binary.Write(data, binary.LittleEndian, sample)
	}

	header := new(bytes.Buffer)
	header.WriteString("RIFF")
	_ = binary.Write(header, binary.LittleEndian, uint32(36+data.Len()))
	header.WriteString("WAVEfmt ")
	_ = binary.Write(header, binary.LittleEndian, uint32(16))
	_ = binary.Write(header, binary.LittleEndian, uint16(1)) // PCM
	_ = binary.Write(header, binary.LittleEndian, uint16(1)) // mono
	_ = binary.Write(header, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(header, binary.LittleEndian, uint32(sampleRate*2)) // byte rate
	_ = binary.Write(header, binary.LittleEndian, uint16(2))            // block align
	_ = binary.Write(header, binary.LittleEndian, uint16(16))           // bits
	header.WriteString("data")
	_ = binary.Write(header, binary.LittleEndian, uint32(data.Len()))
	header.Write(data.Bytes())

	return header.Bytes()
}

func sineS16(samples int) []int16 {
	out := make([]int16, samples)
	for i := range out {
		out[i] = int16(math.Sin(float64(i)*0.1) * 8000)
	}
	return out
}

// transcribeRequest 组一个 multipart 请求，fields 之外的 audio 作为文件上传。
func transcribeRequest(t *testing.T, fields map[string]string, audio []byte, audioType string) *http.Request {
	t.Helper()
	return multipartUpload(t, "/api/v1/transcribe", "audioData", "a.wav", audio, audioType, fields)
}

func TestTranscribeRequiresToken(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	req := transcribeRequest(t, map[string]string{"modelName": testOfflineModel}, nil, "")
	req.Header.Del("Authorization")

	recorder := serveRequest(ts, req)
	assert.Equal(t, http.StatusUnauthorized, recorder.Code)
}

func TestTranscribeRejectsMissingModelName(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, nil, nil, ""))
	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "modelName is required", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsMissingAudio(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{"modelName": testOfflineModel}, nil, ""))
	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "audioData or audioURL is required", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsBadDownMixMethod(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName":     testOfflineModel,
		"downMixMethod": "middle",
	}, wavBytes(testStreamSampleRate, sineS16(16)), "audio/wav"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "invalid down mix method: middle", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsUnsupportedFormat(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
	}, []byte("not audio at all"), "audio/aac"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported audio format", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsCorruptAudio(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
	}, []byte("RIFFxxxxWAVEjunk"), "audio/wav"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "failed to decode audio data", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsUnknownModel(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": "nope",
	}, wavBytes(testStreamSampleRate, sineS16(16)), "audio/wav"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model nope not found", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsNonOfflineModel(t *testing.T) {
	// 只实现了流式接口的推理器不能被离线接口用。
	ts := newAudioTestServer(t, map[string]Inferencer{
		"streamOnly": &fakeStreamingInferencer{sampleRate: testStreamSampleRate},
	}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": "streamOnly",
	}, wavBytes(testStreamSampleRate, sineS16(16)), "audio/wav"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model streamOnly does not support offline asr", recorder.Header().Get("X-Error"))
}

func TestTranscribeFailsWhenInferencerErrors(t *testing.T) {
	ts := newAudioTestServer(t, map[string]Inferencer{
		testOfflineModel: &fakeOfflineInferencer{err: assert.AnError},
	}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
	}, wavBytes(testStreamSampleRate, sineS16(16)), "audio/wav"))

	assert.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.Equal(t, "failed to transcribe", recorder.Header().Get("X-Error"))
}

func TestTranscribeResamplesAndForwardsHotwords(t *testing.T) {
	inferencer := &fakeOfflineInferencer{
		sampleRate: 8000,
		result:     ASRResult{Text: "你好世界", Lang: "zh"},
	}
	ts := newAudioTestServer(t, map[string]Inferencer{testOfflineModel: inferencer}, 0)

	// 输入 16kHz、8000 个采样，模型要 8kHz，重采样后应该大约减半。
	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
		"hotwords":  "天气, 温度",
	}, wavBytes(16000, sineS16(8000)), "audio/wav"))

	require.Equal(t, http.StatusOK, recorder.Code)

	var result ASRResult
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &result))
	assert.Equal(t, "你好世界", result.Text)
	assert.Equal(t, "zh", result.Lang)

	samples, hotwords := inferencer.call()
	assert.Equal(t, "天气, 温度", hotwords)
	assert.InDelta(t, 4000, samples, 64, "16kHz input must be resampled to the model's 8kHz")
}

func TestTranscribeKeepsSampleRateWhenItMatches(t *testing.T) {
	inferencer := &fakeOfflineInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testOfflineModel: inferencer}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
	}, wavBytes(testStreamSampleRate, sineS16(1000)), "audio/wav"))

	require.Equal(t, http.StatusOK, recorder.Code)

	samples, hotwords := inferencer.call()
	assert.Equal(t, 1000, samples, "matching sample rates must not go through resampling")
	assert.Empty(t, hotwords)
}

func TestTranscribeDownmixMethods(t *testing.T) {
	// 立体声：左声道满幅，右声道静音。
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint16(raw[0:], 16000)
	binary.LittleEndian.PutUint16(raw[2:], 0)

	stereoWav := func() []byte {
		data := bytes.Repeat(raw, 100)

		header := new(bytes.Buffer)
		header.WriteString("RIFF")
		_ = binary.Write(header, binary.LittleEndian, uint32(36+len(data)))
		header.WriteString("WAVEfmt ")
		_ = binary.Write(header, binary.LittleEndian, uint32(16))
		_ = binary.Write(header, binary.LittleEndian, uint16(1))
		_ = binary.Write(header, binary.LittleEndian, uint16(2)) // stereo
		_ = binary.Write(header, binary.LittleEndian, uint32(testStreamSampleRate))
		_ = binary.Write(header, binary.LittleEndian, uint32(testStreamSampleRate*4))
		_ = binary.Write(header, binary.LittleEndian, uint16(4))
		_ = binary.Write(header, binary.LittleEndian, uint16(16))
		header.WriteString("data")
		_ = binary.Write(header, binary.LittleEndian, uint32(len(data)))
		header.Write(data)
		return header.Bytes()
	}

	for _, method := range []string{"average", "left", "right"} {
		t.Run(method, func(t *testing.T) {
			ts := newAudioTestServer(t,
				map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

			recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
				"modelName":     testOfflineModel,
				"downMixMethod": method,
			}, stereoWav(), "audio/wav"))

			require.Equal(t, http.StatusOK, recorder.Code)
		})
	}
}

func TestTranscribeRejectsDisallowedAudioURL(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
		"audioURL":  "http://example.com/a.wav",
	}, nil, ""))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "audio URL is not allowed for outgoing requests", recorder.Header().Get("X-Error"))
}

func TestTranscribeRejectsBothAudioSources(t *testing.T) {
	ts := newAudioTestServer(t,
		map[string]Inferencer{testOfflineModel: &fakeOfflineInferencer{}}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
		"audioURL":  "http://example.com/a.wav",
	}, wavBytes(testStreamSampleRate, sineS16(16)), "audio/wav"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t,
		"audioData and audioURL cannot be set at the same time",
		recorder.Header().Get("X-Error"))
}

func TestTranscribeDownloadsAudioFromAllowedURL(t *testing.T) {
	audio := wavBytes(testStreamSampleRate, sineS16(500))
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(audio)
	}))
	t.Cleanup(upstream.Close)

	inferencer := &fakeOfflineInferencer{sampleRate: testStreamSampleRate, result: ASRResult{Text: "url"}}
	ts := newAudioTestServer(t, map[string]Inferencer{testOfflineModel: inferencer}, 0)
	ts.server.config.Outgoing.AllowRegexp = []regexp.Regexp{*regexp.MustCompile(`^` + regexp.QuoteMeta(upstream.URL))}

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
		"audioURL":  upstream.URL + "/a.wav",
	}, nil, ""))

	require.Equal(t, http.StatusOK, recorder.Code)

	samples, _ := inferencer.call()
	assert.Equal(t, 500, samples)
}

func TestTranscribeTrimsHotwords(t *testing.T) {
	inferencer := &fakeOfflineInferencer{sampleRate: testStreamSampleRate}
	ts := newAudioTestServer(t, map[string]Inferencer{testOfflineModel: inferencer}, 0)

	recorder := serveRequest(ts, transcribeRequest(t, map[string]string{
		"modelName": testOfflineModel,
		"hotwords":  "  a , b  ",
	}, wavBytes(testStreamSampleRate, sineS16(10)), "audio/wav"))

	require.Equal(t, http.StatusOK, recorder.Code)

	_, hotwords := inferencer.call()
	assert.Equal(t, "a , b", hotwords, "surrounding spaces are trimmed, the list is not split")
}
