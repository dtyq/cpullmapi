//go:build with_audio || with_image

package cpullmapi

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/argon2"
)

const testToken = "test-token"

// multipartUpload 组一个 multipart POST。fileField 非空时把 file 作为文件字段带上，
// 这样能控制 Content-Type；WriteField 不行，它只发文本字段。
func multipartUpload(
	t *testing.T, path, fileField, filename string, file []byte, fileType string,
	fields map[string]string,
) *http.Request {
	t.Helper()

	body := new(bytes.Buffer)
	writer := multipart.NewWriter(body)
	for key, value := range fields {
		require.NoError(t, writer.WriteField(key, value))
	}
	if file != nil {
		part, err := writer.CreatePart(map[string][]string{
			"Content-Disposition": {fmt.Sprintf(`form-data; name=%q; filename=%q`, fileField, filename)},
			"Content-Type":        {fileType},
		})
		require.NoError(t, err)
		_, err = part.Write(file)
		require.NoError(t, err)
	}
	require.NoError(t, writer.Close())

	req := httptest.NewRequest(http.MethodPost, path, body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Token "+testToken)
	return req
}

func serveRequest(ts *testServer, req *http.Request) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	ts.router.ServeHTTP(recorder, req)
	return recorder
}

// testServer 手工装配一个最小 Server：不起 vips、不加载任何模型，推理器全是假的。
type testServer struct {
	server *Server
	router *gin.Engine
	api    *gin.RouterGroup
}

// newTestServer 只搭骨架并挂上鉴权中间件，业务路由由各 API 的测试自己注册。
func newTestServer(t *testing.T, inferencers map[string]Inferencer, tune func(*Config)) *testServer {
	t.Helper()

	descs := make(map[string]RPDesc[Inferencer], len(inferencers))
	for name, inferencer := range inferencers {
		inferencer := inferencer
		descs[name] = RPDesc[Inferencer]{
			ResourceRequired: 1,
			ObjFactory:       func() (Inferencer, error) { return inferencer, nil },
		}
	}

	memoryPool, err := NewResourcePool(len(descs)+1, descs)
	require.NoError(t, err)

	executor, err := NewExecutorPool(2)
	require.NoError(t, err)
	require.NoError(t, executor.Start(1))
	t.Cleanup(executor.Stop)

	config := Config{HTTP: HTTPConfig{TokenHash: testTokenHash(t, testToken)}}
	if tune != nil {
		tune(&config)
	}

	server := &Server{config: config, executorPool: executor, memoryPool: memoryPool}

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set(ContextKeyRequestID, "test")
		c.Next()
	})
	api := router.Group("/api/v1")
	api.Use(server.checkCredential)

	return &testServer{server: server, router: router, api: api}
}

// testTokenHash 用跟 auth.go 一样的参数算一份 salt+hash，好让请求过鉴权。
func testTokenHash(t *testing.T, token string) []byte {
	t.Helper()

	salt := make([]byte, 16)
	_, err := rand.Read(salt)
	require.NoError(t, err)

	hash := argon2.IDKey([]byte(token), salt, 1, 64*1024, 4, 32)
	return append(salt, hash...)
}
