package cpullmapi

import (
	"context"
	"fmt"
	"os"
	"testing"
)

func TestISNetInferencer(t *testing.T) {
	var err error
	cleanup := initORT(t)
	defer cleanup()

	tcs := []struct {
		name      string
		modelPath string
	}{
		{name: "fp32", modelPath: "./models/onnx-community/ISNet-ONNX/onnx/model.onnx"},
		{name: "fp16", modelPath: "./models/onnx-community/ISNet-ONNX/onnx/model_fp16.onnx"},
		{name: "int8", modelPath: "./models/onnx-community/ISNet-ONNX/onnx/model_int8.onnx"},
		{name: "uint8", modelPath: "./models/onnx-community/ISNet-ONNX/onnx/model_uint8.onnx"},
		{name: "quantized", modelPath: "./models/onnx-community/ISNet-ONNX/onnx/model_quantized.onnx"},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			var inferencer Inferencer
			inferencer, err = NewONNXISNetInferencer(
				ONNXSODCommonConfig{
					ModelPath:              tc.modelPath,
					PreprocessorConfigPath: "./models/onnx-community/ISNet-ONNX/preprocessor_config.json",
				},
			)
			if err != nil {
				t.Fatalf("failed to create onnx inferencer: %v", err)
			}
			defer inferencer.Close()

			image, err := openImage("test/testphoto.jpg")
			if err != nil {
				t.Fatalf("failed to open image: %v", err)
			}

			segments, err := inferencer.SegmentImage(context.Background(), image)
			if err != nil {
				t.Fatalf("failed to segment image: %v", err)
			}

			for i, segment := range segments {
				// image255, err := image.Copy()
				// if err != nil {
				// 	t.Fatalf("failed to copy image: %v", err)
				// }
				// err = image255.DrawRect(
				// 	vips.ColorRGBA{R: 255, G: 255, B: 255, A: 255},
				// 	0, 0, image255.Width(), image255.Height(), true)
				// if err != nil {
				// 	t.Fatalf("failed to draw rect: %v", err)
				// }
				// image255.Close()

				image.AddAlpha()
				err = segment.Multiply(image)
				if err != nil {
					t.Fatalf("failed to multiply image: %v", err)
				}

				bin, _, err := segment.ExportPng(nil)
				if err != nil {
					t.Fatalf("failed to export segment: %v", err)
				}
				err = os.WriteFile(fmt.Sprintf("test/ISNet_%s_segment_%d.png", tc.name, i), bin, 0644)
				if err != nil {
					t.Fatalf("failed to write segment: %v", err)
				}
			}
		})
	}
}
