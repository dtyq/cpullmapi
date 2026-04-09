package cpullmapi

import (
	"crypto/rand"
	"encoding/base32"
	"encoding/base64"
	"fmt"
	"io"
	mathRand "math/rand/v2"
	"net/http"
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
