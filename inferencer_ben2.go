package cpullmapi

import (
	"fmt"
	"image/color"

	goImage "image"

	"github.com/davidbyttow/govips/v2/vips"
)

func ben2PostprocessFunc(in *ONNXSODInferencer, outputArray []float32) ([]*vips.ImageRef, error) {
	gim := goImage.NewRGBA(goImage.Rect(0, 0, in.width, in.height))

	for i := range outputArray {
		// to uint8
		outputArray[i] *= 255
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

func NewBEN2Inferencer(
	modelPath string,
	preprocessorConfigPath string,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		modelPath,
		preprocessorConfigPath,
		"pixel_values",
		"alphas",
		ben2PostprocessFunc,
		onnxSessionOptions,
	)
}
