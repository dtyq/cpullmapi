package cpullmapi

import "github.com/davidbyttow/govips/v2/vips"

type HTTPCode int

const (
	CodeSuccess             HTTPCode = 200
	CodeNotFound            HTTPCode = 404
	CodeUnauthorized        HTTPCode = 401
	CodeForbidden           HTTPCode = 403
	CodeInternalServerError HTTPCode = 500
)

type CommonResponse struct {
	Code    HTTPCode `json:"code"`
	Message string   `json:"message"`
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
