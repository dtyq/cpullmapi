//go:build with_audio

package cpullmapi

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/gin-gonic/gin"
)

const (
	defaultStreamingSampleRate = 16000

	// streamFeedSeconds 是交给后端一块音频的间隔。后端自己还会再切块，这里
	// 只是别让按帧发送的客户端把 Feed 调得过密。
	streamFeedSeconds = 0.1
	streamReadBytes   = 32 * 1024

	streamMaxMessageBytes = 1 << 20
)

type streamingPCMFormat struct {
	SampleRate int
	Channels   int
}

// parseStreamingPCMFormat 解析 audio/L16;rate=16000;channels=1。裸 PCM 没有
// 文件头可猜，参数缺省成 16kHz 单声道。
func parseStreamingPCMFormat(contentType string) (streamingPCMFormat, error) {
	format := streamingPCMFormat{SampleRate: defaultStreamingSampleRate, Channels: 1}
	if strings.TrimSpace(contentType) == "" {
		return format, nil
	}

	parts := strings.Split(contentType, ";")
	switch strings.ToLower(strings.TrimSpace(parts[0])) {
	case "audio/l16", "audio/pcm", "application/octet-stream":
	default:
		return format, fmt.Errorf("unsupported content type %q, want audio/L16", parts[0])
	}

	for _, param := range parts[1:] {
		key, value, found := strings.Cut(strings.TrimSpace(param), "=")
		if !found {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)

		var target *int
		switch key {
		case "rate":
			target = &format.SampleRate
		case "channels":
			target = &format.Channels
		default:
			continue
		}

		parsed, err := strconv.Atoi(value)
		if err != nil || parsed <= 0 {
			return format, fmt.Errorf("invalid %s %q", key, value)
		}
		*target = parsed
	}

	return format, nil
}

// appendS16LE 把 s16le 字节转成 float32 追加到 dst。多声道按算术平均压成单声道。
func appendS16LE(dst []float32, src []byte, channels int) []float32 {
	frame := 2 * channels
	for offset := 0; offset+frame <= len(src); offset += frame {
		var sum int32
		for channel := range channels {
			sum += int32(int16(binary.LittleEndian.Uint16(src[offset+channel*2:])))
		}
		dst = append(dst, float32(sum)/float32(channels)/math.MaxInt16)
	}
	return dst
}

// 流式结果的出口。SSE 和 WebSocket 各实现一个。
type asrStreamSink interface {
	emit(result ASRStreamResult) error
	finish() error
}

// errStreamTimedOut 表示对端一直没有再发数据，跟音频内容无关。
var errStreamTimedOut = errors.New("stream timed out waiting for audio")

const streamSSEDone = "[DONE]"

type sseSink struct {
	writer  gin.ResponseWriter
	flusher http.Flusher
	last    ASRStreamResult
}

func (s *sseSink) emit(result ASRStreamResult) error {
	if !streamResultMoved(s.last, result) {
		return nil
	}
	s.last = result

	payload, err := json.Marshal(result)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.writer, "data: %s\n\n", payload); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

func (s *sseSink) finish() error {
	if _, err := fmt.Fprintf(s.writer, "data: %s\n\n", streamSSEDone); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

type wsSink struct {
	ctx  context.Context
	conn *websocket.Conn
	last ASRStreamResult
}

// wsTranscriptMessage 是服务端推给客户端的唯一一种正常消息。
type wsTranscriptMessage struct {
	Type           string   `json:"type"`
	Text           string   `json:"text"`
	Lang           string   `json:"lang,omitempty"`
	Counter        uint64   `json:"counter,omitempty"`
	Start          *float64 `json:"t0,omitempty"`
	End            *float64 `json:"t1,omitempty"`
	EndOfUtterance bool     `json:"endOfUtterance,omitempty"`
	Final          bool     `json:"final,omitempty"`
}

func (s *wsSink) emit(result ASRStreamResult) error {
	if !streamResultMoved(s.last, result) {
		return nil
	}
	s.last = result

	payload, err := json.Marshal(wsTranscriptMessage{
		Type:           "transcript",
		Text:           result.Text,
		Lang:           result.Lang,
		Counter:        result.Counter,
		Start:          result.Start,
		End:            result.End,
		EndOfUtterance: result.EndOfUtterance,
		Final:          result.Final,
	})
	if err != nil {
		return err
	}
	return s.conn.Write(s.ctx, websocket.MessageText, payload)
}

func (s *wsSink) finish() error { return nil }

// streamResultMoved 判断这次结果值不值得推给客户端。后端每收到一块音频都会
// 返回完整结果，文本没变就没必要重复推。
func streamResultMoved(last, current ASRStreamResult) bool {
	return current.Text != last.Text ||
		current.EndOfUtterance != last.EndOfUtterance ||
		current.Final != last.Final
}

// pcmReader 是一次读取。实现方负责把「对端太久没发数据」映射成 errStreamTimedOut，
// 把正常的音频结束映射成 io.EOF。
type pcmReader func(buf []byte) (int, error)

// driveASRStream 收音频、喂后端、把结果推给 sink。整段会话都在调用方的 goroutine
// 上跑：Feed 会阻塞到后端处理完，中途让出槽位没有意义。
func driveASRStream(stream ASRStream, sampleRate int, read pcmReader, sink asrStreamSink) error {
	blockSize := int(float64(sampleRate) * streamFeedSeconds)
	if blockSize < 1 {
		blockSize = 1
	}

	pending := make([]float32, 0, blockSize*2)
	raw := make([]byte, streamReadBytes)
	var carry []byte

	for {
		n, err := read(raw)
		if n > 0 {
			chunk := raw[:n]
			if len(carry) > 0 {
				chunk = append(carry, chunk...)
				carry = nil
			}

			// 半截采样留到下一轮，别把字节对齐搞乱。
			usable := len(chunk) - len(chunk)%2
			if usable < len(chunk) {
				carry = append([]byte(nil), chunk[usable:]...)
			}
			pending = appendS16LE(pending, chunk[:usable], 1)

			for len(pending) >= blockSize {
				result, err := stream.Feed(pending[:blockSize])
				if err != nil {
					return err
				}
				pending = pending[blockSize:]
				if err := sink.emit(result); err != nil {
					return err
				}
			}
		}

		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
	}

	if len(pending) > 0 {
		result, err := stream.Feed(pending)
		if err != nil {
			return err
		}
		if err := sink.emit(result); err != nil {
			return err
		}
	}

	final, err := stream.Flush()
	if err != nil {
		return err
	}
	if err := sink.emit(final); err != nil {
		return err
	}
	return sink.finish()
}

// pickStreamingInferencer 从池子里拿一个支持流式识别的推理器。槽位由请求的
// ctx 把着，整段会话只占一个。
func (s *Server) pickStreamingInferencer(ctx context.Context, modelName string) (StreamingASRInferencer, error) {
	return pickInferencer[StreamingASRInferencer](s.memoryPool, ctx, modelName, "streaming asr")
}

func (s *Server) streamingASRConfig(c *gin.Context) (ASRStreamConfig, error) {
	config := ASRStreamConfig{Hotwords: strings.TrimSpace(c.Query("hotwords"))}

	if raw := c.Query("rTrimTokens"); raw != "" {
		parsed, err := strconv.ParseUint(raw, 10, 32)
		if err != nil {
			return config, fmt.Errorf("invalid rTrimTokens %q", raw)
		}
		rTrim := uint(parsed)
		config.RTrimTokens = &rTrim
	}
	return config, nil
}

// @BasePath /api/v1

// @Summary transcribe audio as a stream over server-sent events
// @Description Transcribe raw signed 16-bit little-endian PCM, streamed in the request body.
// @Description
// @Description PCM parameters are passed in the Content-Type and the query string rather than as multipart form fields, because streaming has to start handing audio to the model before the body ends and multipart would buffer it first.
// @Description
// @Description The response is a text/event-stream: one "data: {...}" per ASRStreamResult, terminated by "data: [DONE]". Each result carries the full transcript so far, not a delta. Clients must use fetch with a ReadableStream, since EventSource cannot send a request body.
// @Security Token
// @Accept audio/L16
// @Produce text/event-stream
// @Param modelName query string true "model name, for example: LCCStreamingASR"
// @Param hotwords query string false "hotwords, separated by comma"
// @Param rTrimTokens query int false "override the r trim tokens configured for the model"
// @Success 200 {object} ASRStreamResult "one event per result, then data: [DONE]"
// @Failure 400 "bad request"
// @Failure 401 "unauthorized"
// @Header 400 {string} X-Error "error message"
// @Router /transcribe/stream [post]
func (s *Server) transcribeStreamHandler(c *gin.Context) {
	if !c.GetBool(ContextKeyCredentialOK) {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}

	modelName := c.Query("modelName")
	if modelName == "" {
		c.Header("X-Error", "modelName is required")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	format, err := parseStreamingPCMFormat(c.GetHeader("Content-Type"))
	if err != nil {
		c.Header("X-Error", err.Error())
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	config, err := s.streamingASRConfig(c)
	if err != nil {
		c.Header("X-Error", err.Error())
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	inferencer, err := s.pickStreamingInferencer(c.Request.Context(), modelName)
	if err != nil {
		s.Logw("transcribe-stream", "failed to pick inferencer for %s: %v", modelName, err)
		c.Header("X-Error", err.Error())
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// 流式接口不做重采样：状态化的重采样会把「按块喂」的语义搞复杂，直接让
	// 客户端按模型采样率发。
	if format.SampleRate != inferencer.SampleRate() {
		c.Header("X-Error", fmt.Sprintf(
			"model %s wants %d Hz, got %d Hz", modelName, inferencer.SampleRate(), format.SampleRate))
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	stream, err := inferencer.NewASRStream(c.Request.Context(), config)
	if err != nil {
		s.Logw("transcribe-stream", "failed to create stream for %s: %v", modelName, err)
		c.Header("X-Error", "failed to create stream")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}
	defer stream.Close()

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	// 反代别缓冲，否则 SSE 会被攒成一坨。
	c.Header("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Flush()

	sink := &sseSink{writer: c.Writer, flusher: c.Writer}
	read := ssePCMReader(c, s.config.Misc.StreamingASRTimeout)

	done := make(chan error, 1)
	s.executorPool.Dispatch(func() {
		done <- driveASRStream(stream, inferencer.SampleRate(), read, sink)
	})

	if err := <-done; err != nil {
		// 响应头已经发出去了，改不了状态码，只能记日志并在流里断掉。
		s.Logw("transcribe-stream", "stream for %s ended with error: %v", modelName, err)
	}
}

// ssePCMReader 从请求体读 PCM。超时只看对端有没有再发数据，跟音频内容无关。
func ssePCMReader(c *gin.Context, timeout time.Duration) pcmReader {
	body := c.Request.Body
	if timeout <= 0 {
		return body.Read
	}

	controller := http.NewResponseController(c.Writer)
	return func(buf []byte) (int, error) {
		if err := controller.SetReadDeadline(time.Now().Add(timeout)); err != nil {
			// 底层不支持设读超时，那就退化成不设。
			return body.Read(buf)
		}

		n, err := body.Read(buf)
		if errors.Is(err, os.ErrDeadlineExceeded) {
			return n, errStreamTimedOut
		}
		return n, err
	}
}

// wsClientMessage 是客户端发来的控制消息。二进制帧是音频，文本帧是这些。
type wsClientMessage struct {
	Type        string `json:"type"`
	Hotwords    string `json:"hotwords,omitempty"`
	RTrimTokens *uint  `json:"rTrimTokens,omitempty"`
}

// @Summary transcribe audio as a stream over websocket
// @Description Transcribe raw signed 16-bit little-endian PCM sent as binary frames.
// @Description
// @Description The token may be sent either as an "Authorization: Token <token>" header or, for browsers which cannot set headers on a websocket, in the handshake as "Sec-WebSocket-Protocol: realtime, openai-insecure-api-key.<token>".
// @Description
// @Description After the handshake the client may send one optional text frame {"type":"session.update","hotwords":"..."}, then binary PCM frames, and finally {"type":"input_audio_buffer.commit"} to end the audio. The server answers with text frames {"type":"transcript",...}, each carrying the full transcript so far.
// @Security Token
// @Produce application/json
// @Param modelName query string true "model name, for example: LCCStreamingASR"
// @Success 101 {string} string "switching protocols"
// @Failure 400 "bad request"
// @Failure 401 "unauthorized"
// @Header 400 {string} X-Error "error message"
// @Router /transcribe/realtime [get]
func (s *Server) transcribeRealtimeHandler(c *gin.Context) {
	modelName := c.Query("modelName")
	if modelName == "" {
		c.Header("X-Error", "modelName is required")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	if !c.GetBool(ContextKeyCredentialOK) && !s.checkWebsocketCredential(c) {
		c.Header("X-Error", "unauthorized")
		c.AbortWithStatus(http.StatusUnauthorized)
		return
	}

	inferencer, err := s.pickStreamingInferencer(c.Request.Context(), modelName)
	if err != nil {
		s.Logw("transcribe-realtime", "failed to pick inferencer for %s: %v", modelName, err)
		c.Header("X-Error", err.Error())
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	conn, err := websocket.Accept(c.Writer, c.Request, &websocket.AcceptOptions{
		Subprotocols: []string{"realtime"},
		// 鉴权靠 token，不靠 Origin。
		InsecureSkipVerify: true,
	})
	if err != nil {
		s.Logw("transcribe-realtime", "failed to accept websocket: %v", err)
		return
	}
	conn.SetReadLimit(streamMaxMessageBytes)
	defer conn.CloseNow()

	ctx := c.Request.Context()

	// 握手后第一条消息可以是 session.update，用来配这次会话。
	sessionConfig := ASRStreamConfig{}
	client := &wsClient{ctx: ctx, conn: conn}
	if err := client.readOptionalSessionUpdate(&sessionConfig); err != nil {
		s.closeWebsocket(conn, err)
		return
	}

	stream, err := inferencer.NewASRStream(ctx, sessionConfig)
	if err != nil {
		s.Logw("transcribe-realtime", "failed to create stream for %s: %v", modelName, err)
		s.writeWebsocketError(conn, "failed to create stream")
		return
	}
	defer stream.Close()

	sink := &wsSink{ctx: ctx, conn: conn}
	read := client.pcmReader(s.config.Misc.StreamingASRTimeout)

	done := make(chan error, 1)
	s.executorPool.Dispatch(func() {
		done <- driveASRStream(stream, inferencer.SampleRate(), read, sink)
	})

	err = <-done
	if err == nil {
		conn.Close(websocket.StatusNormalClosure, "")
		return
	}

	s.Logw("transcribe-realtime", "stream for %s ended with error: %v", modelName, err)
	s.closeWebsocket(conn, err)
}

func (s *Server) closeWebsocket(conn *websocket.Conn, err error) {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		conn.Close(websocket.StatusNormalClosure, "")
		return
	}
	conn.Close(websocket.StatusInternalError, err.Error())
}

func (s *Server) writeWebsocketError(conn *websocket.Conn, message string) {
	payload, err := json.Marshal(map[string]string{"type": "error", "message": message})
	if err != nil {
		return
	}
	_ = conn.Write(context.Background(), websocket.MessageText, payload)
}

// checkWebsocketCredential 从子协议里取 token。浏览器只能这么传。
func (s *Server) checkWebsocketCredential(c *gin.Context) bool {
	const prefix = "openai-insecure-api-key."

	for _, protocol := range c.Request.Header.Values("Sec-WebSocket-Protocol") {
		for _, candidate := range strings.Split(protocol, ",") {
			token, found := strings.CutPrefix(strings.TrimSpace(candidate), prefix)
			if found && s.verifyToken(token) {
				return true
			}
		}
	}
	return false
}

type wsClient struct {
	ctx  context.Context
	conn *websocket.Conn

	// 一次 Read 拿到一整条消息，但调用方按固定大小读，读剩的留在这里。
	pending []byte
}

func (c *wsClient) readOptionalSessionUpdate(config *ASRStreamConfig) error {
	ctx, cancel := context.WithTimeout(c.ctx, 30*time.Second)
	defer cancel()

	typ, data, err := c.conn.Read(ctx)
	if err != nil {
		return err
	}
	if typ != websocket.MessageText {
		// 直接就开始发音频了，那第一条留在 pending 里给后面的读取。
		c.pending = data
		return nil
	}

	var message wsClientMessage
	if err := json.Unmarshal(data, &message); err != nil {
		return fmt.Errorf("invalid session message: %w", err)
	}
	if message.Type != "session.update" {
		return fmt.Errorf("expected session.update, got %q", message.Type)
	}

	config.Hotwords = strings.TrimSpace(message.Hotwords)
	config.RTrimTokens = message.RTrimTokens
	return nil
}

// pcmReader 交出一条按需读取的 PCM 流。commit 等价于音频结束。
func (c *wsClient) pcmReader(timeout time.Duration) pcmReader {
	return func(buf []byte) (int, error) {
		for len(c.pending) == 0 {
			ctx := c.ctx
			if timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}

			typ, data, err := c.conn.Read(ctx)
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					return 0, errStreamTimedOut
				}

				// 对端正常收尾，或者链路断了，都不会再有音频了，当成音频结束。
				switch websocket.CloseStatus(err) {
				case websocket.StatusNormalClosure, websocket.StatusGoingAway, -1:
					return 0, io.EOF
				}

				if c.ctx.Err() != nil {
					return 0, c.ctx.Err()
				}
				return 0, err
			}

			switch typ {
			case websocket.MessageBinary:
				c.pending = data
			case websocket.MessageText:
				var message wsClientMessage
				if err := json.Unmarshal(data, &message); err != nil {
					return 0, fmt.Errorf("invalid control message: %w", err)
				}
				if message.Type == "input_audio_buffer.commit" {
					return 0, io.EOF
				}
				return 0, fmt.Errorf("unexpected control message %q", message.Type)
			}
		}

		n := copy(buf, c.pending)
		c.pending = c.pending[n:]
		return n, nil
	}
}
