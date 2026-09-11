//go:build with_image

package cpullmapi_test

// The config tests look factories up by model type, and the backends are what
// register them. Nothing else pulls a backend into this test binary.
import (
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"

	_ "github.com/dtyq/cpullmapi/backends/onnx"
)

const onnxRuntimeLib = "libs/onnxruntime/lib/libonnxruntime.so"

// 根包里有几个用例要真的建出 ONNX 推理器，得先把运行时库挂上。
func TestMain(m *testing.M) {
	if _, err := os.Stat(onnxRuntimeLib); err == nil {
		ort.SetSharedLibraryPath(onnxRuntimeLib)
		ort.InitializeEnvironment()
	}

	code := m.Run()
	ort.DestroyEnvironment()
	os.Exit(code)
}
