//go:build with_image

package onnx

import (
	"testing"

	"github.com/davidbyttow/govips/v2/vips"
	ort "github.com/yalue/onnxruntime_go"
)

func openImage(imagePath string) (*vips.ImageRef, error) {
	image, err := vips.NewImageFromFile(imagePath)
	if err != nil {
		return nil, err
	}
	return image, nil
}

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
