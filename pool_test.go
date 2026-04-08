package cpullmapi

import (
	"sync"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestExecutorPool(t *testing.T) {
	pool, err := NewExecutorPool(2)
	if err != nil {
		t.Fatalf("failed to create executor pool: %v", err)
	}

	err = pool.Start(2)
	if err != nil {
		t.Fatalf("failed to start executor pool: %v", err)
	}
	defer pool.Stop()

	ort.SetSharedLibraryPath(onnxSharedLibraryPath)

	err = ort.InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize ort environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	var wg sync.WaitGroup
	wg.Add(1)
	testFunc := func() {
		defer wg.Done()

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

		for _, segment := range segments {
			_, _, err := segment.ExportPng(nil)
			if err != nil {
				t.Fatalf("failed to export segment: %v", err)
			}
		}
	}

	pool.Dispatch(testFunc)
	wg.Wait()
}
