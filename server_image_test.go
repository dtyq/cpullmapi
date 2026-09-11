//go:build with_image

package cpullmapi

import (
	"sync"
	"testing"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/stretchr/testify/require"
)

const testMattingModel = "MattingModel"

const (
	testMattingWidth  = 8
	testMattingHeight = 6
)

// vips 只在 image.go 的 init 里注册了启动函数，正常路径由 Config.MiscInitialize
// 触发；测试不走那条路，这里自己起一次。Startup 重复调用是安全的。
var vipsOnce sync.Once

func requireVips(t *testing.T) {
	t.Helper()

	var err error
	vipsOnce.Do(func() { err = vips.Startup(nil) })
	require.NoError(t, err)
}

func newImageTestServer(t *testing.T, inferencers map[string]Inferencer) *testServer {
	t.Helper()

	requireVips(t)

	ts := newTestServer(t, inferencers, nil)
	ts.api.POST("/matting", ts.server.mattingHandler)
	return ts
}
