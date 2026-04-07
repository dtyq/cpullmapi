package cpullmapi

import (
	"fmt"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestBiRefNetInferencer(t *testing.T) {
	ort.SetSharedLibraryPath(onnxSharedLibraryPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize ort environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	tcs := []struct {
		name      string
		modelPath string
	}{
		{name: "fp32", modelPath: "./models/onnx-community/BiRefNet-ONNX/onnx/model.onnx"},
		{name: "fp16", modelPath: "./models/onnx-community/BiRefNet-ONNX/onnx/model_fp16.onnx"},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			var inferencer Inferencer
			inferencer, err = NewBiRefNetInferencer(
				"./models/onnx-community/BiRefNet-ONNX/onnx/model_fp16.onnx",
				"./models/onnx-community/BiRefNet-ONNX/preprocessor_config.json",
			)
			if err != nil {
				t.Fatalf("failed to create onnx inferencer: %v", err)
			}
			defer inferencer.Close()

			image, err := openImage("test/testphoto.jpg")
			if err != nil {
				t.Fatalf("failed to open image: %v", err)
			}

			segments, err := inferencer.SegmentImage(image)
			if err != nil {
				t.Fatalf("failed to segment image: %v", err)
			}

			for i, segment := range segments {
				bin, _, err := segment.ExportPng(nil)
				if err != nil {
					t.Fatalf("failed to export segment: %v", err)
				}
				err = os.WriteFile(fmt.Sprintf("test/BiRefNet_%s_segment_%d.png", tc.name, i), bin, 0644)
				if err != nil {
					t.Fatalf("failed to write segment: %v", err)
				}
			}

		})
	}
}
