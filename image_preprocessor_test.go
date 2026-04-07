package cpullmapi

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path"
	"strings"
	"testing"
)

const binetImageProcessorConfigJSON = `
{
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "feature_extractor_type": "ViTFeatureExtractor",
  "image_mean": [
    0.485,
    0.456,
    0.406
  ],
  "image_processor_type": "ViTFeatureExtractor",
  "image_std": [
    0.229,
    0.224,
    0.225
  ],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 1024,
    "width": 1024
  }
}
`

func TestViTImageProcessor(t *testing.T) {
	t.SkipNow()
	testImages := []string{
		"test/photo.jpg",
		"test/qwen.webp",
		"test/firecloud.png",
		"test/elysia.bmp",
	}
	for _, testImage := range testImages {
		image, err := openImage(testImage)
		if err != nil {
			t.Fatalf("failed to open image: %v", err)
		}

		var config ViTImageProcessorConfig
		err = json.Unmarshal([]byte(binetImageProcessorConfigJSON), &config)
		if err != nil {
			t.Fatalf("failed to unmarshal image processor config: %v", err)
		}

		processor, err := NewViTImageProcessor(config)
		if err != nil {
			t.Fatalf("failed to create image processor: %v", err)
		}

		chwArray, err := processor.PreprocessImage(image)
		if err != nil {
			t.Fatalf("failed to preprocess image: %v", err)
		}

		extName := path.Ext(testImage)
		outputPath := strings.TrimSuffix(testImage, extName) + ".bin"
		fo, err := os.OpenFile(outputPath, os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			t.Fatalf("failed to open output file: %v", err)
		}
		defer fo.Close()

		for _, value := range chwArray {
			binary.Write(fo, binary.NativeEndian, value)
		}
	}
}
