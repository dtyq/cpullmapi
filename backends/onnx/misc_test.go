//go:build with_image

package onnx

import (
	"testing"

	"github.com/davidbyttow/govips/v2/vips"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/dtyq/cpullmapi/internal/testutil"
)

const (
	onnxRuntimeLib = "./lib/onnxruntime_openvino/onnxruntime/capi/libonnxruntime.so.1.24.1"
	onnxModelsDir  = "../../models/onnx-community"
	onnxTestImage  = "../../test/testphoto.jpg"
	onnxTestOutDir = "../../test"
)

func openImage(imagePath string) (*vips.ImageRef, error) {
	return vips.NewImageFromFile(imagePath)
}

func initORT(t *testing.T) func() {
	testutil.RequireFiles(t, onnxRuntimeLib)

	ort.SetSharedLibraryPath(onnxRuntimeLib)

	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("failed to initialize ort environment: %v", err)
	}
	return func() {
		ort.DestroyEnvironment()
	}
}
