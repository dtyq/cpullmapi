//go:build with_image

package onnx

import (
	"context"
	"fmt"
	"os"
	"testing"

	core "github.com/dtyq/cpullmapi"
	"github.com/dtyq/cpullmapi/internal/testutil"
)

func TestISNetInferencer(t *testing.T) {
	var err error
	cleanup := initORT(t)
	defer cleanup()

	preprocessorConfigPath := onnxModelsDir + "/ISNet-ONNX/preprocessor_config.json"
	tcs := []struct {
		name      string
		modelPath string
	}{
		{name: "fp32", modelPath: onnxModelsDir + "/ISNet-ONNX/onnx/model.onnx"},
		{name: "fp16", modelPath: onnxModelsDir + "/ISNet-ONNX/onnx/model_fp16.onnx"},
		{name: "int8", modelPath: onnxModelsDir + "/ISNet-ONNX/onnx/model_int8.onnx"},
		{name: "uint8", modelPath: onnxModelsDir + "/ISNet-ONNX/onnx/model_uint8.onnx"},
		{name: "quantized", modelPath: onnxModelsDir + "/ISNet-ONNX/onnx/model_quantized.onnx"},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			testutil.RequireFiles(t, tc.modelPath, preprocessorConfigPath, onnxTestImage)

			var inferencer core.ImageSegmentationInferencer
			inferencer, err = NewONNXISNetInferencer(
				ONNXSODCommonConfig{
					ModelPath:              tc.modelPath,
					PreprocessorConfigPath: preprocessorConfigPath,
				},
			)
			if err != nil {
				t.Fatalf("failed to create onnx inferencer: %v", err)
			}
			defer inferencer.Close()

			image, err := openImage(onnxTestImage)
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
				err = os.WriteFile(fmt.Sprintf("%s/ISNet_%s_segment_%d.png", onnxTestOutDir, tc.name, i), bin, 0644)
				if err != nil {
					t.Fatalf("failed to write segment: %v", err)
				}
			}
		})
	}
}
