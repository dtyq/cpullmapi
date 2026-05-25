//go:build with_image

package cpullmapi

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/davidbyttow/govips/v2/vips"
)

type ImagePostprocessFunc[T any] func(in *T, outputArray []float32) ([]*vips.ImageRef, error)

func InplaceResizeImage(image *vips.ImageRef, width int, height int, resizeMethod PILResampleMethod) error {
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

func vipsRGBAFromHTMLHex(str string, rgba *vips.ColorRGBA) error {
	str = strings.TrimPrefix(str, "#")

	if len(str) != 8 && len(str) != 6 {
		return fmt.Errorf("invalid rgba hex color: %s", str)
	}

	var err error
	var u64 uint64
	u64, err = strconv.ParseUint(str[0:2], 16, 8)
	if err != nil {
		return fmt.Errorf("failed to parse rgba hex color: %w", err)
	}
	rgba.R = uint8(u64)
	u64, err = strconv.ParseUint(str[2:4], 16, 8)
	if err != nil {
		return fmt.Errorf("failed to parse rgba hex color: %w", err)
	}
	rgba.G = uint8(u64)
	u64, err = strconv.ParseUint(str[4:6], 16, 8)
	if err != nil {
		return fmt.Errorf("failed to parse rgba hex color: %w", err)
	}
	rgba.B = uint8(u64)
	if len(str) == 8 {
		u64, err = strconv.ParseUint(str[6:8], 16, 8)
		if err != nil {
			return fmt.Errorf("failed to parse rgba hex color: %w", err)
		}
		rgba.A = uint8(u64)
	} else {
		rgba.A = 255
	}
	return nil
}

type ImageMIME string

const (
	ImageMIMEJPEG ImageMIME = "image/jpeg"
	ImageMIMEPNG  ImageMIME = "image/png"
	ImageMIMEBMP  ImageMIME = "image/bmp"
	ImageMIMEGIF  ImageMIME = "image/gif"
	ImageMIMEWebP ImageMIME = "image/webp"

	ImageMIMETIFF ImageMIME = "image/tiff"
	ImageMIMEJP2K ImageMIME = "image/jp2"
	ImageMIMEJXL  ImageMIME = "image/jxl"
	ImageMIMEHEIF ImageMIME = "image/heif"
)

func (mime ImageMIME) ToVipsImageType() vips.ImageType {
	switch mime {
	case ImageMIMEJPEG:
		return vips.ImageTypeJPEG
	case ImageMIMEPNG:
		return vips.ImageTypePNG
	case ImageMIMEBMP:
		return vips.ImageTypeBMP
	case ImageMIMEGIF:
		return vips.ImageTypeGIF
	case ImageMIMEWebP:
		return vips.ImageTypeWEBP
	case ImageMIMETIFF:
		return vips.ImageTypeTIFF
	case ImageMIMEJP2K:
		return vips.ImageTypeJP2K
	case ImageMIMEJXL:
		return vips.ImageTypeJXL
	case ImageMIMEHEIF:
		return vips.ImageTypeHEIF
	default:
		return vips.ImageTypeUnknown
	}
}

func (mime *ImageMIME) FromVipsImageType(imageType vips.ImageType) {
	switch imageType {
	case vips.ImageTypeJPEG:
		*mime = ImageMIMEJPEG
	case vips.ImageTypePNG:
		*mime = ImageMIMEPNG
	case vips.ImageTypeBMP:
		*mime = ImageMIMEBMP
	case vips.ImageTypeGIF:
		*mime = ImageMIMEGIF
	case vips.ImageTypeWEBP:
		*mime = ImageMIMEWebP
	case vips.ImageTypeTIFF:
		*mime = ImageMIMETIFF
	case vips.ImageTypeJP2K:
		*mime = ImageMIMEJP2K
	case vips.ImageTypeJXL:
		*mime = ImageMIMEJXL
	case vips.ImageTypeHEIF:
		*mime = ImageMIMEHEIF
	}
}

func init() {
	ConfigInitFuncs = append(ConfigInitFuncs, func(c *Config) error {
		var err error

		// initialize vips
		vips.LoggingSettings(func(messageDomain string, messageLevel vips.LogLevel, message string) {
			// empty implementation to omit startup logs
		}, c.Inference.VIPSLoggingLevel)
		err = vips.Startup(&vips.Config{
			ConcurrencyLevel: c.Inference.ThreadsPerSlot,
		})
		if err != nil {
			return fmt.Errorf("failed to initialize vips: %v", err)
		}

		return err
	})

	ConfigShutdownFuncs = append(ConfigShutdownFuncs, func(c *Config) {
		// shutdown vips
		vips.Shutdown()
	})
}
