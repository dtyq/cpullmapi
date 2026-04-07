package cpullmapi

import (
	"fmt"

	"github.com/davidbyttow/govips/v2/vips"
)

func openImage(imagePath string) (*vips.ImageRef, error) {
	image, err := vips.NewImageFromFile(imagePath)
	if err != nil {
		return nil, err
	}
	return image, nil
}

func inplaceResizeImage(image *vips.ImageRef, width int, height int, resizeMethod PILResampleMethod) error {
	var err error
	widthFactor := float64(width) / float64(image.Width())
	heightFactor := float64(height) / float64(image.Height())
	switch resizeMethod {
	case PILResampleMethodNearest:
		err = image.ResizeWithVScale(widthFactor, heightFactor, vips.KernelNearest)
	case PILResampleMethodLANCZOS:
		err = image.ResizeWithVScale(widthFactor, heightFactor, vips.KernelLanczos3)
	case PILResampleMethodBilinear:
		err = image.ResizeWithVScale(widthFactor, heightFactor, vips.KernelLinear)
	case PILResampleMethodBicubic:
		err = image.ResizeWithVScale(widthFactor, heightFactor, vips.KernelCubic)
	default:
		return fmt.Errorf("not implemented resize method: %d", resizeMethod)
	}
	return err
}

func inplaceImageToRGBU8Array(image *vips.ImageRef, backgroundColor *vips.Color) ([]byte, error) {
	var err error

	// convert image to sRGB space
	err = image.ToColorSpace(vips.InterpretationSRGB)
	if err != nil {
		return nil, err
	}

	// white as background
	if image.HasAlpha() {
		if backgroundColor == nil {
			backgroundColor = &vips.Color{R: 255, G: 255, B: 255}
		}
		err = image.Flatten(backgroundColor)
		if err != nil {
			return nil, err
		}
	}

	err = image.Cast(vips.BandFormatUchar)
	if err != nil {
		return nil, err
	}

	rgbArray, err := image.ToBytes()
	if err != nil {
		return nil, err
	}
	return rgbArray, nil
}
