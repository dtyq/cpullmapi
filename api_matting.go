package cpullmapi

import (
	"io"
	"net/http"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/gin-gonic/gin"
)

// @BasePath /api/v1

// @Summary matting image
// @Description matting image, output image format is determined by the Accept header, if accept is not set, use the input image format
// @Security Token
// @Produce image/jpeg,image/png,image/bmp,image/gif,image/webp,image/tiff,image/jp2,image/jxl,image/heif
// @Param X-Model header string true "model name, for example: MVANet"
// @Param imageData formData file false "image data in binary, conflicts with imageURL"
// @Param imageURL formData string false "image url, conflicts with imageData"
// @Param backgroundColor formData string false "background color in rgba hex format, for example: #ffffffff, default is #ffffff00 for white"
// @Success 200 {file} file "image data in binary"
// @Failure 401 "unauthorized"
// @Failure 400 "bad request"
// @Header 400 {string} X-Error "error message"
// @Failure 500 "internal server error"
// @Router /matting [post]
func (s *Server) mattingHandler(c *gin.Context) {
	var err error

	if !c.GetBool(ContextKeyCredentialOK) {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}

	// check model name
	modelName := c.GetHeader("X-Model")
	_, ok := s.config.Inference.Models[modelName]
	if !ok {
		c.Header("X-Error", "unknown model")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	backgroundColor := vips.ColorRGBA{
		R: 255,
		G: 255,
		B: 255,
		A: 0,
	}
	if c.PostForm("backgroundColor") != "" {
		err = vipsRGBAFromHTMLHex(c.PostForm("backgroundColor"), &backgroundColor)
		if err != nil {
			s.Logw("matting", "failed to parse background color: %v", err)
			c.Header("X-Error", "failed to parse background color")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
	}

	// get image url and form data
	imageURL := c.PostForm("imageURL")
	imageDataForm, err := c.FormFile("imageData")
	if err != nil {
		imageDataForm = nil
	}
	if imageDataForm == nil && imageURL == "" {
		c.Header("X-Error", "imageData or imageURL is required")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}
	if imageDataForm != nil && imageURL != "" {
		c.Header("X-Error", "imageData and imageURL cannot be set at the same time")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// read image data from form or url
	var imageDataFile io.ReadCloser
	if imageDataForm != nil {
		imageDataFile, err = imageDataForm.Open()
		if err != nil {
			c.Header("X-Error", "failed to read image data from form")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
	} else {
		allowed := false
		for _, re := range s.config.Outgoing.AllowRegexp {
			if re.MatchString(imageURL) {
				allowed = true
				break
			}
		}
		if !allowed {
			s.Logw("matting", "image URL is not allowed for outgoing requests: %s", imageURL)
			c.Header("X-Error", "image URL is not allowed for outgoing requests")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		// download image from url
		req, err := http.NewRequestWithContext(c.Request.Context(), "GET", imageURL, nil)
		if err != nil {
			s.Logw("matting", "failed to create request for image URL: %v", err)
			c.Header("X-Error", "failed to create request for image URL")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			s.Logw("matting", "failed to download image from URL: %s: %v", imageURL, err)
			c.Header("X-Error", "failed to download image from URL")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
		imageDataFile = resp.Body
	}
	defer imageDataFile.Close()

	// read image data from file
	imageData, err := io.ReadAll(imageDataFile)
	if err != nil {
		s.Logw("matting", "failed to read image data: %v", err)
		c.Header("X-Error", "failed to read image data")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// open image
	image, err := vips.NewImageFromBuffer(imageData)
	if err != nil {
		s.Logw("matting", "failed to open image from buffer: %v", err)
		c.Header("X-Error", "failed to open image from buffer")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}
	defer image.Close()

	if image.Bands() == 1 {
		// convert to sRGB
		err = image.ToColorSpace(vips.InterpretationSRGB)
		if err != nil {
			s.Logw("matting", "failed to convert image to sRGB: %v", err)
			c.Header("X-Error", "failed to convert image to sRGB")
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
	if image.Bands() != 3 && image.Bands() != 4 {
		s.Logw("matting", "image channels is not 3 or 4: %d", image.Bands())
		c.Header("X-Error", "image channels is not 3 or 4")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}
	if image.BandFormat() != vips.BandFormatFloat {
		// cast to float
		err = image.Cast(vips.BandFormatFloat)
		if err != nil {
			s.Logw("matting", "failed to cast image to float: %v", err)
			c.Header("X-Error", "failed to cast image to float")
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}

	// determine output image format
	accept := ImageMIME(c.GetHeader("Accept"))
	outputVIPSFormat := accept.ToVipsImageType()
	if outputVIPSFormat == vips.ImageTypeUnknown {
		// use input image format
		outputVIPSFormat = image.Format()
	}

	// do inference
	channel := make(chan *vips.ImageRef)
	s.executorPool.Dispatch(func() {
		inferencer, err := s.memoryPool.GetObj(c.Request.Context(), modelName)
		if err != nil {
			s.Logw("matting", "failed to get inferencer: %v", err)
			close(channel)
			return
		}

		segments, err := inferencer.SegmentImage(c.Request.Context(), image)
		if err != nil {
			s.Logw("matting", "failed to segment image: %v", err)
			close(channel)
			return
		}

		// for matting api, we only use the first segment
		channel <- segments[0]
		close(channel)
	})

	segment := <-channel
	if segment == nil {
		c.Header("X-Error", "failed to segment image")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}

	// to apply mul, add alpha channel
	if !image.HasAlpha() {
		// add alpha channel
		err = image.AddAlpha()
		if err != nil {
			s.Logw("matting", "failed to add alpha channel: %v", err)
			c.Header("X-Error", "failed to add alpha channel")
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
	// debug 255 image
	// err = image.DrawRect(
	// 	vips.ColorRGBA{R: 255, G: 255, B: 255, A: 255},
	// 	0, 0, image.Width(), image.Height(), true)
	// if err != nil {
	// 	s.Logw("matting", "failed to draw rect: %v", err)
	// 	c.Header("X-Error", "failed to draw rect")
	// 	c.AbortWithStatus(http.StatusInternalServerError)
	// 	return
	// }
	// resize segment to image size
	err = inplaceResizeImage(segment, image.Width(), image.Height(), PILResampleMethodBilinear)
	if err != nil {
		s.Logw("matting", "failed to resize segment: %v", err)
		c.Header("X-Error", "failed to resize segment")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}
	// multiply image and segment to get the matting image
	err = image.Multiply(segment)
	if err != nil {
		s.Logw("matting", "failed to postprocess image: %v", err)
		c.Header("X-Error", "failed to postprocess image")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}

	switch outputVIPSFormat {
	case vips.ImageTypeJPEG:
		// jpeg do not support alpha channel, flatten to background color
		fallthrough
	case vips.ImageTypeBMP:
		// assume bmp is 3 channels
		err := image.Flatten(&vips.Color{
			R: backgroundColor.R,
			G: backgroundColor.G,
			B: backgroundColor.B,
		})
		if err != nil {
			s.Logw("matting", "failed to flatten image to background color: %v", err)
			c.Header("X-Error", "failed to flatten image to background color")
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	default:
		if backgroundColor.A != 0 ||
			backgroundColor.R != 255 ||
			backgroundColor.G != 255 ||
			backgroundColor.B != 255 {
			// not trival background color, apply background color to image
			imBytes := make([]byte, image.Width()*image.Height()*4)
			for y := 0; y < image.Height(); y++ {
				for x := 0; x < image.Width(); x++ {
					imBytes[y*image.Width()*4+x*4+0] = backgroundColor.R
					imBytes[y*image.Width()*4+x*4+1] = backgroundColor.G
					imBytes[y*image.Width()*4+x*4+2] = backgroundColor.B
					imBytes[y*image.Width()*4+x*4+3] = backgroundColor.A
				}
			}
			bgImage, err := vips.NewImageFromHWCArray(imBytes, 4, image.Width(), image.Height(), vips.BandFormatUchar, vips.InterpretationSRGB)
			if err != nil {
				s.Logw("matting", "failed to create background image: %v", err)
				c.Header("X-Error", "failed to create background image")
				c.AbortWithStatus(http.StatusInternalServerError)
				return
			}

			err = bgImage.Cast(image.BandFormat())
			if err != nil {
				bgImage.Close()
				s.Logw("matting", "failed to cast background image to image band format: %v", err)
				c.Header("X-Error", "failed to cast background image to image band format")
				c.AbortWithStatus(http.StatusInternalServerError)
				return
			}

			// fmt.Printf("bgImage band format: %d\n", bgImage.BandFormat())
			// fmt.Printf("image band format: %d\n", image.BandFormat())
			// fmt.Printf("bgImage colorspace: %d\n", bgImage.ColorSpace())
			// fmt.Printf("image colorspace: %d\n", image.ColorSpace())

			err = bgImage.Composite(image, vips.BlendModeOver, 0, 0)
			if err != nil {
				bgImage.Close()
				s.Logw("matting", "failed to composite background image: %v", err)
				c.Header("X-Error", "failed to composite background image")
				c.AbortWithStatus(http.StatusInternalServerError)
				return
			}

			image.Close()
			image = bgImage
		}
	}
	var exportFunc func(*vips.ImageRef) ([]byte, error)
	switch outputVIPSFormat {
	case vips.ImageTypeJPEG:
		exportFunc = func(im *vips.ImageRef) ([]byte, error) {
			data, _, err := im.ExportJpeg(&vips.JpegExportParams{
				Quality: 81, // TODO: custom quality
			})
			return data, err
		}
	case vips.ImageTypePNG:
		exportFunc = func(im *vips.ImageRef) ([]byte, error) {
			data, _, err := im.ExportPng(&vips.PngExportParams{
				Compression: 6, // TODO: custom compression level
			})
			return data, err
		}
	case vips.ImageTypeWEBP:
		exportFunc = func(im *vips.ImageRef) ([]byte, error) {
			data, _, err := im.ExportWebp(&vips.WebpExportParams{
				Quality: 75, // TODO: custom quality
			})
			return data, err
		}
	case vips.ImageTypeTIFF:
		exportFunc = func(im *vips.ImageRef) ([]byte, error) {
			data, _, err := im.ExportTiff(&vips.TiffExportParams{
				Compression: vips.TiffCompressionLzw, // TODO: custom compression
			})
			return data, err
		}
	default:
		exportFunc = func(im *vips.ImageRef) ([]byte, error) {
			data, _, err := im.Export(&vips.ExportParams{
				Format: outputVIPSFormat,
			})
			return data, err
		}
	}

	newImageData, err := exportFunc(image)
	if err != nil {
		s.Logw("matting", "failed to export image: %v", err)
		c.Header("X-Error", "failed to export image")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}
	var outputMime ImageMIME
	outputMime.FromVipsImageType(outputVIPSFormat)
	c.Data(http.StatusOK, string(outputMime), newImageData)
}
