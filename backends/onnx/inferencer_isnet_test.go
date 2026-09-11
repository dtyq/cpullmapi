//go:build with_image

package onnx

import (
	"context"
	"fmt"
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
				writeSegmentFiles(t, image, segment, fmt.Sprintf("ISNet_%s", tc.name), i)
			}
		})
	}
}
