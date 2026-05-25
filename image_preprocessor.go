//go:build with_image

package cpullmapi

import (
	"fmt"

	"github.com/davidbyttow/govips/v2/vips"
)

type PILResampleMethod int

const (
	PILResampleMethodNearest  PILResampleMethod = 0
	PILResampleMethodLANCZOS  PILResampleMethod = 1
	PILResampleMethodBilinear PILResampleMethod = 2
	PILResampleMethodBicubic  PILResampleMethod = 3
	PILResampleMethodBox      PILResampleMethod = 4
	PILResampleMethodHAMMING  PILResampleMethod = 5
)

type ViTImageProcessorConfig struct {
	DoNormalize   *bool              `json:"do_normalize"`
	DoRescale     *bool              `json:"do_rescale"`
	DoResize      *bool              `json:"do_resize"`
	ImageMean     *[]float32         `json:"image_mean"`
	ImageStd      *[]float32         `json:"image_std"`
	Resample      *PILResampleMethod `json:"resample"`
	RescaleFactor *float32           `json:"rescale_factor"`
	Size          struct {
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"size"`
}

type ViTImageProcessor struct {
	config ViTImageProcessorConfig
}

func NewViTImageProcessor(config ViTImageProcessorConfig) (*ViTImageProcessor, error) {
	return &ViTImageProcessor{
		config: config,
	}, nil
}

func (p *ViTImageProcessor) PreprocessImage(image *vips.ImageRef) ([]float32, error) {
	var err error

	var resizeMethod PILResampleMethod
	var rescaleFactor float32
	var mean [3]float32
	var std [3]float32

	// 1. resize image to config.Size
	if p.config.DoResize != nil && *p.config.DoResize == false {
		goto skipResize
	}

	resizeMethod = PILResampleMethodBilinear
	if p.config.Resample != nil {
		resizeMethod = *p.config.Resample
	}
	err = inplaceResizeImage(image, p.config.Size.Width, p.config.Size.Height, resizeMethod)
	if err != nil {
		return nil, err
	}

skipResize:

	// pre-2 convert image to RGB array
	rgbU8Array, err := inplaceImageToRGBU8Array(image, nil)
	if err != nil {
		return nil, err
	}
	// image reference is now pointing to the resized and converted (to rgb888) image
	// so just imagine that image is freed
	hwcArray := make([]float32, len(rgbU8Array))
	for i := range rgbU8Array {
		hwcArray[i] = float32(rgbU8Array[i])
	}

	// 2. rescale pixel values
	if p.config.DoRescale != nil && *p.config.DoRescale == false {
		goto skipRescale
	}

	rescaleFactor = float32(1) / float32(255)
	if p.config.RescaleFactor != nil {
		rescaleFactor = *p.config.RescaleFactor
	}
	for i := range hwcArray {
		hwcArray[i] = hwcArray[i] * rescaleFactor
	}
skipRescale:

	// 3. normalize(-mean, /std) pixel values
	if p.config.DoNormalize != nil && *p.config.DoNormalize == false {
		goto skipNormalize
	}

	mean = [3]float32{0.5, 0.5, 0.5}
	std = [3]float32{0.5, 0.5, 0.5}
	if p.config.ImageMean != nil {
		if len(*p.config.ImageMean) != 3 {
			return nil, fmt.Errorf("image mean must be a list of 3 values")
		}
		mean = [3]float32{
			(*p.config.ImageMean)[0],
			(*p.config.ImageMean)[1],
			(*p.config.ImageMean)[2],
		}
	}
	if p.config.ImageStd != nil {
		if len(*p.config.ImageStd) != 3 {
			return nil, fmt.Errorf("image std must be a list of 3 values")
		}
		if (*p.config.ImageStd)[0] == 0 {
			return nil, fmt.Errorf("image std[0] is 0")
		}
		if (*p.config.ImageStd)[1] == 0 {
			return nil, fmt.Errorf("image std[1] is 0")
		}
		if (*p.config.ImageStd)[2] == 0 {
			return nil, fmt.Errorf("image std[2] is 0")
		}
		std = [3]float32{
			(*p.config.ImageStd)[0],
			(*p.config.ImageStd)[1],
			(*p.config.ImageStd)[2],
		}
	}

	for i := range len(hwcArray) / 3 {
		hwcArray[i*3] = (hwcArray[i*3] - mean[0]) / std[0]
		hwcArray[i*3+1] = (hwcArray[i*3+1] - mean[1]) / std[1]
		hwcArray[i*3+2] = (hwcArray[i*3+2] - mean[2]) / std[2]
	}

skipNormalize:

	// convert HWC to CHW
	chwArray := make([]float32, len(hwcArray))
	for i := range len(hwcArray) / 3 {
		chwArray[i] = hwcArray[i*3]
		chwArray[i+len(hwcArray)/3] = hwcArray[i*3+1]
		chwArray[i+len(hwcArray)/3*2] = hwcArray[i*3+2]
	}

	return chwArray, nil
}
