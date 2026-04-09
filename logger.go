package cpullmapi

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mattn/go-isatty"
)

type LogStreamConfig struct {
	Level LogLevel `yaml:"level"`
	Kind  LogKind  `yaml:"kind"`
	Type  LogType  `yaml:"type"`
	Color *bool    `yaml:"color,omitempty"` // true to enable color, false to disable color, default to check isatty
	Path  string   `yaml:"path,omitempty"`
}

type LogKind string

const (
	LogKindFile LogKind = "file"
	// LogKindSLS  LogKind = "sls"
)

type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarning
	LogLevelError
)

type LogType string

const (
	LogTypeText LogType = "text"
	LogTypeJSON LogType = "json"
)

type Logger interface {
	Log(level LogLevel, tag string, logEntry LogEntry)
}

func (s *Server) openLogStreams() error {
	var err error
	for _, config := range s.config.Logs {
		switch config.Kind {
		case LogKindFile:
			var stream *os.File
			switch {
			case config.Path == "stdout":
				stream = os.Stdout
			case config.Path == "stderr":
				stream = os.Stderr
			case strings.HasPrefix(config.Path, "/") || strings.HasPrefix(config.Path, "."):
				stream, err = os.OpenFile(config.Path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
				if err != nil {
					return fmt.Errorf("failed to open log file: %w", err)
				}
			default:
				return fmt.Errorf("unsupported log path (use relative path starting with . or absolute path starting with /): %s", config.Path)
			}
			logger := &FileLogger{
				stream: stream,
				config: config,
			}
			if config.Color != nil {
				logger.color = *config.Color
			} else {
				logger.color = isatty.IsTerminal(stream.Fd())
			}
			s.loggers = append(s.loggers, logger)
		default:
			return fmt.Errorf("unsupported log kind: %s", config.Kind)
		}
	}
	return nil
}

type FileLogger struct {
	stream io.WriteCloser
	config LogStreamConfig
	color  bool
}

var logColors = map[LogLevel]string{
	LogLevelDebug:   "\033[96m",
	LogLevelInfo:    "",
	LogLevelWarning: "\033[33;1m",
	LogLevelError:   "\033[31;1m",
}

var logLevelNames = map[LogLevel]string{
	LogLevelDebug:   "D",
	LogLevelInfo:    "I",
	LogLevelWarning: "W",
	LogLevelError:   "E",
}

type LogEntry interface {
	ToMap() map[string]any
	ToString() string
}

type TextLogEntry struct {
	str string
}

func (l TextLogEntry) ToMap() map[string]any {
	return map[string]any{
		"msg": l.str,
	}
}

func (l TextLogEntry) ToString() string {
	return l.str
}

func (l FileLogger) Log(level LogLevel, tag string, logEntry LogEntry) {
	if level < l.config.Level {
		return
	}

	switch l.config.Type {
	case LogTypeText:
		// prepare color and level name
		color := ""
		if l.color {
			color = logColors[level]
		}
		levelName := logLevelNames[level]
		dateString := time.Now().Format("2006-01-02 15:04:05.000")
		logString := logEntry.ToString()

		// write to stream
		for _, line := range strings.Split(logString, "\n") {
			fmt.Fprintf(l.stream, "%s[%s][%s][%s]\033[0m %s\n", color, dateString, levelName, tag, line)
		}
	case LogTypeJSON:
		jsonDict := map[string]any{}
		for key, value := range logEntry.ToMap() {
			jsonDict[key] = value
		}
		jsonDict["date"] = time.Now().Format("2006-01-02 15:04:05.000")
		jsonDict["lv"] = level
		jsonDict["tag"] = tag
		jsonBytes, _ := json.Marshal(jsonDict)
		l.stream.Write(append(jsonBytes, []byte("\n")...))
	}
}

func (s Server) Log(level LogLevel, tag string, logEntry LogEntry) {
	for _, logger := range s.loggers {
		logger.Log(level, tag, logEntry)
	}
}

func (s Server) Logd(tag string, format string, args ...any) {
	s.Log(LogLevelDebug, tag, TextLogEntry{fmt.Sprintf(format, args...)})
}

func (s Server) Logi(tag string, format string, args ...any) {
	s.Log(LogLevelInfo, tag, TextLogEntry{fmt.Sprintf(format, args...)})
}

func (s Server) Logw(tag string, format string, args ...any) {
	s.Log(LogLevelWarning, tag, TextLogEntry{fmt.Sprintf(format, args...)})
}

func (s Server) Loge(tag string, format string, args ...any) {
	s.Log(LogLevelError, tag, TextLogEntry{fmt.Sprintf(format, args...)})
}

type GinAccessLogEntry struct {
	requestID  string
	clientIP   string
	method     string
	path       string
	query      string
	statusCode int
	latency    time.Duration
	bodySize   int
}

func (l GinAccessLogEntry) ToMap() map[string]any {
	return map[string]any{
		"requestID":  l.requestID,
		"clientIP":   l.clientIP,
		"method":     l.method,
		"path":       l.path,
		"query":      l.query,
		"statusCode": l.statusCode,
		"latency":    l.latency.String(),
		"bodySize":   l.bodySize,
	}
}

func (l GinAccessLogEntry) ToString() string {
	return fmt.Sprintf("%s %s %s%s %d %s %d", l.clientIP, l.method, l.path, l.query, l.statusCode, l.latency, l.bodySize)
}

func (s Server) ginLogger(c *gin.Context) {
	start := c.GetTime(ContextKeyRequestStartTime)
	// fmt.Println("start", start)
	requestID := c.GetString(ContextKeyRequestID)
	if requestID == "" {
		requestID = "-"
	}
	path := c.Request.URL.Path
	raw := c.Request.URL.RawQuery
	if raw != "" {
		raw = "?" + raw
	}

	c.Next()

	// Stop timer
	now := time.Now()
	latency := now.Sub(start)

	clientIP := c.ClientIP()
	method := c.Request.Method
	statusCode := c.Writer.Status()
	bodySize := c.Writer.Size()

	logEntry := &GinAccessLogEntry{
		requestID:  requestID,
		clientIP:   clientIP,
		method:     method,
		path:       path,
		query:      raw,
		statusCode: statusCode,
		latency:    latency,
		bodySize:   bodySize,
	}

	s.Log(LogLevelInfo, requestID, logEntry)
}
