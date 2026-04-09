package cpullmapi

import (
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

const onnxSharedLibraryPath = "./lib/onnxruntime_openvino/onnxruntime/capi/libonnxruntime.so.1.24.1"

func initORT(t *testing.T) func() {
	ort.SetSharedLibraryPath(onnxSharedLibraryPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize ort environment: %v", err)
	}
	return func() {
		ort.DestroyEnvironment()
	}
}
