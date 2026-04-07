package cpullmapi

import (
	"fmt"
	"image/color"
	"math"

	goImage "image"

	"github.com/davidbyttow/govips/v2/vips"
)

func birefnetPostprocessFunc(in *ONNXSODInferencer, outputArray []float32) ([]*vips.ImageRef, error) {
	gim := goImage.NewRGBA(goImage.Rect(0, 0, in.width, in.height))

	for i := range outputArray {
		// sigmoid, mul 255
		x := float64(outputArray[i])
		outputArray[i] = float32(1/(1+math.Exp(-x))) * 255
		// to uint8
		gim.SetRGBA(
			i%in.width,
			i/in.width,
			color.RGBA{
				uint8(outputArray[i]),
				uint8(outputArray[i]),
				uint8(outputArray[i]),
				uint8(outputArray[i]),
			},
		)
	}
	segment, err := vips.NewImageFromGoImage(gim)
	if err != nil {
		return nil, fmt.Errorf("failed to create image from go image: %v", err)
	}

	return []*vips.ImageRef{segment}, nil
}

func NewBiRefNetInferencer(
	modelPath string,
	preprocessorConfigPath string,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		modelPath,
		preprocessorConfigPath,
		"input_image",
		"output_image",
		birefnetPostprocessFunc,
		onnxSessionOptions,
	)
}
