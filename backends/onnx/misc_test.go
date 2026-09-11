//go:build with_image

package onnx

import (
	"fmt"
	"os"
	"testing"

	"github.com/davidbyttow/govips/v2/vips"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/dtyq/cpullmapi/internal/testutil"
)

const (
	onnxRuntimeLib = "../../libs/onnxruntime/lib/libonnxruntime.so"
	onnxModelsDir  = "../../models/onnx-community"
	onnxTestImage  = "../../play/testphoto.jpg"
	onnxTestOutDir = "../../play"
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

// writeSegmentFiles 存两张图供肉眼检查：mask 本身，和抠出来的图。
func writeSegmentFiles(t *testing.T, image, segment *vips.ImageRef, name string, index int) {
	t.Helper()

	writeMask(t, segment, fmt.Sprintf("%s/%s_mask_%d.png", onnxTestOutDir, name, index))
	writeCutout(t, image, segment, fmt.Sprintf("%s/%s_segment_%d.png", onnxTestOutDir, name, index))
}

// writeMask 导出 mask 灰度图。mask 是 0..1 的浮点，直接导出会退化成全黑，先拉到 0..255。
func writeMask(t *testing.T, segment *vips.ImageRef, path string) {
	t.Helper()

	mask, err := segment.Copy()
	if err != nil {
		t.Fatalf("failed to copy segment: %v", err)
	}
	defer mask.Close()

	if err := mask.Linear([]float64{255}, []float64{0}); err != nil {
		t.Fatalf("failed to scale segment: %v", err)
	}
	if err := mask.Cast(vips.BandFormatUchar); err != nil {
		t.Fatalf("failed to cast segment: %v", err)
	}

	bin, _, err := mask.ExportPng(nil)
	if err != nil {
		t.Fatalf("failed to export mask: %v", err)
	}
	if err := os.WriteFile(path, bin, 0o644); err != nil {
		t.Fatalf("failed to write mask: %v", err)
	}
}

// writeCutout 把 mask 当 alpha 乘到原图上，跟 matting 接口的产出一致。
func writeCutout(t *testing.T, image, segment *vips.ImageRef, path string) {
	t.Helper()

	cutout, err := image.Copy()
	if err != nil {
		t.Fatalf("failed to copy image: %v", err)
	}
	defer cutout.Close()

	if !cutout.HasAlpha() {
		if err := cutout.AddAlpha(); err != nil {
			t.Fatalf("failed to add alpha channel: %v", err)
		}
	}
	if err := cutout.Multiply(segment); err != nil {
		t.Fatalf("failed to multiply image: %v", err)
	}

	bin, _, err := cutout.ExportPng(nil)
	if err != nil {
		t.Fatalf("failed to export cutout: %v", err)
	}
	if err := os.WriteFile(path, bin, 0o644); err != nil {
		t.Fatalf("failed to write cutout: %v", err)
	}
}
