//go:build with_audio

package cpullmapi

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeASRStream 记下喂进来的音频，按脚本吐结果。
type fakeASRStream struct {
	mu      sync.Mutex
	fed     [][]float32
	flushed bool
	closed  bool

	// 第 n 次 Feed 返回脚本里的第 n 个结果。
	results []ASRStreamResult
	err     error
}

func (s *fakeASRStream) Feed(samples []float32) (ASRStreamResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.fed = append(s.fed, append([]float32(nil), samples...))
	if s.err != nil {
		return ASRStreamResult{}, s.err
	}
	if len(s.results) == 0 {
		return ASRStreamResult{}, nil
	}
	result := s.results[0]
	s.results = s.results[1:]
	return result, nil
}

func (s *fakeASRStream) Flush() (ASRStreamResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.flushed = true
	if s.err != nil {
		return ASRStreamResult{}, s.err
	}
	return ASRStreamResult{Text: "final", Final: true}, nil
}

func (s *fakeASRStream) Reset() error { return nil }
func (s *fakeASRStream) Close()       { s.mu.Lock(); s.closed = true; s.mu.Unlock() }

func (s *fakeASRStream) fedSamples() []float32 {
	s.mu.Lock()
	defer s.mu.Unlock()

	var all []float32
	for _, chunk := range s.fed {
		all = append(all, chunk...)
	}
	return all
}

func (s *fakeASRStream) isFlushed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.flushed
}

func (s *fakeASRStream) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

// fakeStreamingInferencer 只实现流式接口。
type fakeStreamingInferencer struct {
	sampleRate int
	stream     *fakeASRStream
	config     ASRStreamConfig

	mu           sync.Mutex
	streamErrors []error
}

func (i *fakeStreamingInferencer) GetCapabilities() []Capability {
	return []Capability{CapabilityStreamingASR}
}

func (i *fakeStreamingInferencer) Close() {}

func (i *fakeStreamingInferencer) SampleRate() int { return i.sampleRate }

func (i *fakeStreamingInferencer) streamConfig() ASRStreamConfig {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.config
}

func (i *fakeStreamingInferencer) NewASRStream(_ context.Context, config ASRStreamConfig) (ASRStream, error) {
	i.mu.Lock()
	defer i.mu.Unlock()

	i.config = config
	if len(i.streamErrors) > 0 {
		err := i.streamErrors[0]
		i.streamErrors = i.streamErrors[1:]
		return nil, err
	}
	if i.stream == nil {
		i.stream = &fakeASRStream{}
	}
	return i.stream, nil
}

// pcmBytes 生成一段 16bit 单声道 PCM。
func pcmBytes(samples int) []byte {
	buf := make([]byte, samples*2)
	for i := range samples {
		value := int16(math.Sin(float64(i)*0.1) * 8000)
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(value))
	}
	return buf
}

func TestParseStreamingPCMFormat(t *testing.T) {
	tcs := []struct {
		name        string
		contentType string
		expect      streamingPCMFormat
		expectErr   string
	}{
		{name: "empty", contentType: "", expect: streamingPCMFormat{SampleRate: 16000, Channels: 1}},
		{name: "bare", contentType: "audio/L16", expect: streamingPCMFormat{SampleRate: 16000, Channels: 1}},
		{
			name: "with params", contentType: "audio/L16;rate=8000;channels=2",
			expect: streamingPCMFormat{SampleRate: 8000, Channels: 2},
		},
		{
			name: "spaces and case", contentType: "Audio/PCM; RATE = 44100 ; Channels = 1 ",
			expect: streamingPCMFormat{SampleRate: 44100, Channels: 1},
		},
		{name: "octet stream", contentType: "application/octet-stream", expect: streamingPCMFormat{SampleRate: 16000, Channels: 1}},
		{name: "bad type", contentType: "audio/wav", expectErr: "unsupported content type"},
		{name: "bad rate", contentType: "audio/L16;rate=abc", expectErr: "invalid rate"},
		{name: "zero channels", contentType: "audio/L16;channels=0", expectErr: "invalid channels"},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			format, err := parseStreamingPCMFormat(tc.contentType)
			if tc.expectErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.expect, format)
		})
	}
}

func TestAppendS16LE(t *testing.T) {
	// 三个采样：满幅正、满幅负、零。
	raw := make([]byte, 6)
	binary.LittleEndian.PutUint16(raw[0:], math.MaxInt16)
	// -32768 的补码。
	binary.LittleEndian.PutUint16(raw[2:], 0x8000)
	binary.LittleEndian.PutUint16(raw[4:], 0)

	got := appendS16LE(nil, raw, 1)
	require.Len(t, got, 3)
	assert.InDelta(t, 1.0, got[0], 0.001)
	assert.InDelta(t, -1.0, got[1], 0.001)
	assert.InDelta(t, 0.0, got[2], 0.001)

	// 半截采样（3 字节）只应解出 1 个。
	assert.Len(t, appendS16LE(nil, raw[:3], 1), 1)
}

func TestAppendS16LEDownmixes(t *testing.T) {
	// 立体声：左右分别是满幅和零，平均后应为半幅。
	raw := make([]byte, 4)
	binary.LittleEndian.PutUint16(raw[0:], uint16(32767))
	binary.LittleEndian.PutUint16(raw[2:], 0)

	got := appendS16LE(nil, raw, 2)
	require.Len(t, got, 1)
	assert.InDelta(t, 0.5, got[0], 0.001)
}

func TestDriveASRStreamFeedsAllAudio(t *testing.T) {
	stream := &fakeASRStream{}
	sink := &recordingSink{}

	// 16000 Hz、100ms 一块就是 1600 个采样，给 5000 个采样。
	raw := pcmBytes(5000)
	read := bytesReader(raw)

	require.NoError(t, driveASRStream(stream, 16000, read, sink))

	fed := stream.fedSamples()
	assert.Equal(t, 5000, len(fed), "every sample should reach the backend")
	assert.True(t, stream.isFlushed(), "Flush should be called once at the end")
	assert.True(t, sink.finished, "sink should be finished")
	assert.InDelta(t, 0, fed[0], 1e-6, "the first sample is a zero crossing")
	assert.NotZero(t, fed[100], "later samples must carry real audio")
}

func TestDriveASRStreamHandlesOddByteBoundary(t *testing.T) {
	stream := &fakeASRStream{}
	sink := &recordingSink{}

	raw := pcmBytes(100)
	// 拆成奇数字节的分片，逼出半截采样的处理。
	var chunks [][]byte
	for i := 0; i < len(raw); i += 3 {
		end := min(i+3, len(raw))
		chunks = append(chunks, raw[i:end])
	}

	require.NoError(t, driveASRStream(stream, 16000, chunksReader(chunks), sink))
	assert.Equal(t, 100, len(stream.fedSamples()), "odd splits must not lose or duplicate samples")
}

func TestDriveASRStreamStopsOnTimeout(t *testing.T) {
	stream := &fakeASRStream{}
	sink := &recordingSink{}

	var calls int
	read := func(buf []byte) (int, error) {
		calls++
		if calls == 1 {
			n := copy(buf, pcmBytes(10))
			return n, nil
		}
		return 0, errStreamTimedOut
	}

	err := driveASRStream(stream, 16000, read, sink)
	assert.ErrorIs(t, err, errStreamTimedOut)
	assert.False(t, stream.isFlushed(), "a timeout must not look like end of audio")
}

func TestDriveASRStreamPropagatesBackendError(t *testing.T) {
	stream := &fakeASRStream{err: assert.AnError}
	sink := &recordingSink{}

	err := driveASRStream(stream, 16000, bytesReader(pcmBytes(10)), sink)
	assert.ErrorIs(t, err, assert.AnError)
}

// recordingSink 收集推出去的结果。
type recordingSink struct {
	mu       sync.Mutex
	results  []ASRStreamResult
	finished bool
}

func (s *recordingSink) emit(result ASRStreamResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.results = append(s.results, result)
	return nil
}

func (s *recordingSink) finish() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finished = true
	return nil
}

func bytesReader(raw []byte) pcmReader {
	offset := 0
	return func(buf []byte) (int, error) {
		if offset >= len(raw) {
			return 0, io.EOF
		}
		n := copy(buf, raw[offset:])
		offset += n
		return n, nil
	}
}

func chunksReader(chunks [][]byte) pcmReader {
	index := 0
	return func(buf []byte) (int, error) {
		if index >= len(chunks) {
			return 0, io.EOF
		}
		chunk := chunks[index]
		index++
		return copy(buf, chunk), nil
	}
}
