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

func TestBEN2Inferencer(t *testing.T) {
	var err error
	cleanup := initORT(t)
	defer cleanup()

	tcs := []struct {
		name                   string
		modelPath              string
		preprocessorConfigPath string
	}{
		{
			name:                   "BEN2",
			modelPath:              onnxModelsDir + "/BEN2-ONNX/onnx/model_fp16.onnx",
			preprocessorConfigPath: onnxModelsDir + "/BEN2-ONNX/preprocessor_config.json",
		},
		{
			name:                   "ORMBG",
			modelPath:              onnxModelsDir + "/ormbg-ONNX/onnx/model_fp16.onnx",
			preprocessorConfigPath: onnxModelsDir + "/ormbg-ONNX/preprocessor_config.json",
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			testutil.RequireFiles(t, tc.modelPath, tc.preprocessorConfigPath, onnxTestImage)

			var inferencer core.ImageSegmentationInferencer
			inferencer, err = NewONNXBEN2Inferencer(
				ONNXSODCommonConfig{
					ModelPath:              tc.modelPath,
					PreprocessorConfigPath: tc.preprocessorConfigPath,
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
				bin, _, err := segment.ExportPng(nil)
				if err != nil {
					t.Fatalf("failed to export segment: %v", err)
				}
				err = os.WriteFile(fmt.Sprintf("%s/%s_segment_%d.png", onnxTestOutDir, tc.name, i), bin, 0644)
				if err != nil {
					t.Fatalf("failed to write segment: %v", err)
				}
			}
		})
	}
}
