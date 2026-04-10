package cpullmapi

import (
	"crypto/rand"
	"encoding/base32"
	"encoding/base64"
	"fmt"
	"io"
	mathRand "math/rand/v2"
	"net"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/gin-gonic/gin"
	swaggerfiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	"golang.org/x/sys/unix"

	"gopkg.in/yaml.v3"

	"github.com/dtyq/cpullmapi/docs"
)

type HTTPConfig struct {
	Bind      string `yaml:"bind"`
	Debug     bool   `yaml:"debug"`
	TokenHash []byte `yaml:"tokenHash"`

	TLSCert string `yaml:"tlsCert"`
	TLSKey  string `yaml:"tlsKey"`
}

func (c *HTTPConfig) UnmarshalYAML(value *yaml.Node) error {
	var tmp struct {
		Bind      string `yaml:"bind"`
		Debug     bool   `yaml:"debug"`
		TokenHash string `yaml:"tokenHash"`
		TLSCert   string `yaml:"tlsCert"`
		TLSKey    string `yaml:"tlsKey"`
	}

	if err := value.Decode(&tmp); err != nil {
		return err
	}

	c.Bind = tmp.Bind
	c.Debug = tmp.Debug
	c.TLSCert = tmp.TLSCert
	c.TLSKey = tmp.TLSKey

	decodedTokenHash, err := base64.StdEncoding.DecodeString(tmp.TokenHash)
	if err != nil {
		return fmt.Errorf("failed to decode token hash: %v", err)
	}
	if len(decodedTokenHash) != 48 {
		return fmt.Errorf("token hash must be 48 bytes binary base64ed")
	}
	c.TokenHash = decodedTokenHash

	// check tls cert and key
	if c.TLSCert != "" {
		if err := unix.Access(c.TLSCert, unix.O_RDONLY); err != nil {
			return fmt.Errorf("failed to access tls cert: %v", err)
		}
	}
	if c.TLSKey != "" {
		if err := unix.Access(c.TLSKey, unix.O_RDONLY); err != nil {
			return fmt.Errorf("failed to access tls key: %v", err)
		}
	}

	return nil
}

type Server struct {
	config Config

	loggers   []Logger
	notCSPRNG io.Reader

	// for http
	router *gin.Engine

	// for inferencer
	executorPool ExecutorPool
	memoryPool   *ResourcePool[Inferencer]
}

type ContextKeyRequestStartTimeType struct{}
type ContextKeyRequestIDType struct{}

var (
	ContextKeyRequestStartTime = ContextKeyRequestStartTimeType{}
	ContextKeyRequestID        = ContextKeyRequestIDType{}
)

func (c Config) CreateServer() (*Server, error) {
	executorPool, err := NewExecutorPool(c.Inference.SlotCount)
	if err != nil {
		return nil, fmt.Errorf("failed to create executor pool: %v", err)
	}

	memoryPool, err := c.CreateMemoryPool()
	if err != nil {
		return nil, fmt.Errorf("failed to create memory pool: %v", err)
	}

	seedBuf := [32]byte{}
	_, err = rand.Read(seedBuf[:])
	if err != nil {
		return nil, fmt.Errorf("failed to read not-CSPRNG seed: %w", err)
	}
	notCSPRNG := mathRand.NewChaCha8(seedBuf)

	s := &Server{
		config:       c,
		notCSPRNG:    notCSPRNG,
		executorPool: executorPool,
		memoryPool:   memoryPool,
	}

	if err := s.openLogStreams(); err != nil {
		return nil, fmt.Errorf("failed to open log streams: %v", err)
	}

	// setup vips logging
	vips.LoggingSettings(func(messageDomain string, messageLevel vips.LogLevel, message string) {
		switch messageLevel {
		case vips.LogLevelDebug:
			s.Logd("vips-"+messageDomain, "%s", message)
		case vips.LogLevelInfo:
			s.Logi("vips-"+messageDomain, "%s", message)
		case vips.LogLevelWarning:
			s.Logw("vips-"+messageDomain, "%s", message)
		case vips.LogLevelError:
			s.Loge("vips-"+messageDomain, "%s", message)
		}
	}, s.config.Inference.VIPSLoggingLevel)

	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		requestStartTime := time.Now()
		c.Set(ContextKeyRequestStartTime, requestStartTime)

		buf := make([]byte, 16)
		s.notCSPRNG.Read(buf)
		// binary.NativeEndian.PutUint16(buf, uint16(requestStartTime.UnixMilli()))
		requestID := base32.StdEncoding.EncodeToString(buf[:15])
		c.Set(ContextKeyRequestID, requestID)
		c.Writer.Header().Set("X-Request-ID", requestID)

		c.Next()
	})
	if s.config.Incoming.ProxyRequestInspector {
		router.Use(s.proxyRequestInspector)
	}
	if len(s.config.Incoming.AllowCIDR) > 0 {
		router.Use(s.checkCIDR)
	}
	router.Use(s.ginLogger)

	// api group
	apiGroup := router.Group("/api/v1")
	apiGroup.Use(s.checkCredential)
	// apiGroup.Use(disableHTTPAPIAccess)
	apiGroup.GET("/healthcheck", s.healthcheck)
	// inference
	apiGroup.POST("/matting", s.mattingHandler)

	// swagger
	if s.config.HTTP.Debug {
		docs.SwaggerInfo.BasePath = "/api/v1"
		router.Use(func(c *gin.Context) {
			if c.Request.URL.Path == "/api/swagger/" {
				c.Redirect(http.StatusMovedPermanently, "/api/swagger/index.html")
				return
			}
			c.Next()
		})
		router.GET("/api/swagger/*any", ginSwagger.WrapHandler(swaggerfiles.Handler))
	}

	router.NoRoute(func(c *gin.Context) {
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1") {
			c.AbortWithStatusJSON(http.StatusNotFound, gin.H{
				"code":    CodeNotFound,
				"message": "not found",
			})
		} else {
			// TODO: jump to a html 404 page
		}
	})

	s.router = router
	return s, nil
}

func (s *Server) Run() error {
	err := s.executorPool.Start(s.config.Inference.ThreadsPerSlot)
	if err != nil {
		return fmt.Errorf("failed to start executor pool: %v", err)
	}
	defer s.executorPool.Stop()

	if s.config.HTTP.TLSCert != "" && s.config.HTTP.TLSKey != "" {
		err = http.ListenAndServeTLS(s.config.HTTP.Bind, s.config.HTTP.TLSCert, s.config.HTTP.TLSKey, s.router)
		if err != nil {
			return fmt.Errorf("failed to run http server: %v", err)
		}
	} else {
		err = http.ListenAndServe(s.config.HTTP.Bind, s.router)
		if err != nil {
			return fmt.Errorf("failed to run http server: %v", err)
		}
	}

	return nil
}

func (s *Server) proxyRequestInspector(c *gin.Context) {
	if c.Request.Method != http.MethodConnect && c.Request.URL.Scheme == "" {
		c.Next()
		return
	}

	requestID := c.GetString(ContextKeyRequestID)

	// do inspect
	remoteAddress := c.Request.RemoteAddr
	s.Logd(requestID+"-connect-inspector", "remoteAddr: %s", remoteAddress)
	for field, values := range c.Request.Header {
		for i, value := range values {
			s.Logd(requestID+"-connect-inspector", "header[%d]: %s: %s", i, field, value)
		}
	}

	hj, ok := c.Writer.(http.Hijacker)
	if !ok {
		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
			"code":    418,
			"message": "I'm a teapot",
		})
		return
	}
	conn, _, err := hj.Hijack()
	if err != nil {
		return
	}
	_ = conn.Close()
}

var bombs = map[string][]byte{}

func (s *Server) checkCIDR(c *gin.Context) {
	requestID := c.GetString(ContextKeyRequestID)

	var encodings []string
	var remoteIP net.IP

	host, _, err := net.SplitHostPort(c.Request.RemoteAddr)
	if err != nil {
		goto reset
	}
	remoteIP = net.ParseIP(host)
	for _, cidr := range s.config.Incoming.AllowCIDR {
		if cidr.Contains(remoteIP) {
			c.Next()
			return
		}
	}
	// do inspect
	s.Logd(requestID+"-compression-bomb", "remoteAddr: %s", c.Request.RemoteAddr)
	for field, values := range c.Request.Header {
		for i, value := range values {
			s.Logd(requestID+"-compression-bomb", "header[%d]: %s: %s", i, field, value)
		}
	}

	if c.GetHeader("Accept-Encoding") == "" {
		goto reset
	}

	// make compression bombs
	encodings = strings.Split(c.GetHeader("Accept-Encoding"), ",")
	// strip spaces
	for i := range len(encodings) {
		encodings[i] = strings.TrimSpace(encodings[i])
	}
	// sort encodings by dict order
	sort.Slice(encodings, func(i, j int) bool {
		dict := map[string]int{
			"deflate": 4,
			"gzip":    3,
			"zstd":    2,
			"br":      1,
		}
		dictI, _ := dict[encodings[i]]
		dictJ, _ := dict[encodings[j]]
		return dictI < dictJ
	})
	for _, encoding := range encodings {
		var err error
		bomb, ok := bombs[encoding]
		if ok {
			s.Logd(requestID+"-compression-bomb", "bomb using existing: %s", encoding)
			c.Header("Content-Type", "text/html")
			c.Header("Content-Encoding", encoding)
			c.Header("Content-Length", fmt.Sprintf("%d", len(bomb)))
			c.Writer.WriteHeader(http.StatusOK)
			c.Writer.Write(bomb)
			c.Writer.Flush()
			c.Abort()
			return
		}
		switch encoding {
		case "gzip":
			bomb, err = os.ReadFile("bomb.gzip")
			if err != nil {
				continue
			}
			bombs[encoding] = bomb
		case "deflate":
			bomb, err = os.ReadFile("bomb.deflate")
			if err != nil {
				continue
			}
			bombs[encoding] = bomb
		case "br":
			bomb, err = os.ReadFile("bomb.br")
			if err != nil {
				continue
			}
			bombs[encoding] = bomb
		case "zstd":
			bomb, err = os.ReadFile("bomb.zstd")
			if err != nil {
				continue
			}
			bombs[encoding] = bomb
		}
		s.Logd(requestID+"-compression-bomb", "bomb using new: %s", encoding)
		c.Header("Content-Type", "text/html")
		c.Header("Content-Encoding", encoding)
		c.Header("Content-Length", fmt.Sprintf("%d", len(bomb)))
		c.Writer.WriteHeader(http.StatusOK)
		c.Writer.Write(bomb)
		c.Writer.Flush()
		c.Abort()
		return
	}

reset:
	hj, ok := c.Writer.(http.Hijacker)
	if !ok {
		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
			"code":    418,
			"message": "I'm a teapot",
		})
		return
	}
	conn, _, err := hj.Hijack()
	if err != nil {
		return
	}
	_ = conn.Close()
}
