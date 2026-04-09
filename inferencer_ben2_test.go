package cpullmapi

import (
	"context"
	"fmt"
	"os"
	"testing"
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
			modelPath:              "./models/onnx-community/BEN2-ONNX/onnx/model_fp16.onnx",
			preprocessorConfigPath: "./models/onnx-community/BEN2-ONNX/preprocessor_config.json",
		},
		{
			name:                   "ORMBG",
			modelPath:              "./models/onnx-community/ormbg-ONNX/onnx/model_fp16.onnx",
			preprocessorConfigPath: "./models/onnx-community/ormbg-ONNX/preprocessor_config.json",
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			var inferencer Inferencer
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

			image, err := openImage("test/testphoto.jpg")
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
				err = os.WriteFile(fmt.Sprintf("test/%s_segment_%d.png", tc.name, i), bin, 0644)
				if err != nil {
					t.Fatalf("failed to write segment: %v", err)
				}
			}
		})
	}
}
