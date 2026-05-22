package cpullmapi

import (
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/gopxl/beep/v2"
	"github.com/gopxl/beep/v2/flac"
	"github.com/gopxl/beep/v2/mp3"
	"github.com/gopxl/beep/v2/vorbis"
	"github.com/gopxl/beep/v2/wav"
)

// @BasePath /api/v1

// @Summary transcribe audio
// @Description transcribe audio, output text and language,
// @Security Token
// @Produce application/json
// @Param modelName formData string true "model name, for example: FireRedASR"
// @Param downMixMethod formData string false "down mix method, for example: average(default), left, right"
// @Param hotwords formData string false "hotwords, separated by comma"
// @Param audioData formData file false "audio data in WAVE format, conflicts with audioURL"
// @Param audioURL formData string false "audio url, conflicts with audioData"
// @Success 200 {object} ASRResult "ASR result"
// @Failure 401 "unauthorized"
// @Failure 400 "bad request"
// @Header 400 {string} X-Error "error message"
// @Failure 500 "internal server error"
// @Router /transcribe [post]
func (s *Server) transcribeHandler(c *gin.Context) {
	var err error

	if !c.GetBool(ContextKeyCredentialOK) {
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"code":    CodeUnauthorized,
			"message": "unauthorized",
		})
		return
	}

	// check model name
	modelName := c.PostForm("modelName")
	if modelName == "" {
		c.Header("X-Error", "modelName is required")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// check model name
	hotwords := c.PostForm("hotwords")
	hotwords = strings.TrimSpace(hotwords)

	// down mix method
	downMixMethod := c.PostForm("downMixMethod")
	switch downMixMethod {
	case "average":
	case "left":
	case "right":
	case "":
		downMixMethod = "average"
	default:
		c.Header("X-Error", "invalid down mix method: "+downMixMethod)
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// get audio url and form data
	audioURL := c.PostForm("audioURL")
	audioDataForm, err := c.FormFile("audioData")
	if err != nil {
		audioDataForm = nil
	}
	if audioDataForm == nil && audioURL == "" {
		c.Header("X-Error", "audioData or audioURL is required")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}
	if audioDataForm != nil && audioURL != "" {
		c.Header("X-Error", "audioData and audioURL cannot be set at the same time")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// read image data from form or url
	inputMIME := "audio/wav"
	var audioDataFile io.ReadCloser
	if audioDataForm != nil {
		inputMIME = audioDataForm.Header.Get("Content-Type")
		if inputMIME != "" {
			inputMIME = strings.TrimSpace(strings.Split(inputMIME, ";")[0])
		}
		audioDataFile, err = audioDataForm.Open()
		if err != nil {
			c.Header("X-Error", "failed to read audio data from form")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
	} else {
		allowed := false
		for _, re := range s.config.Outgoing.AllowRegexp {
			if re.MatchString(audioURL) {
				allowed = true
				break
			}
		}
		if !allowed {
			s.Logw("transcribe", "audio URL is not allowed for outgoing requests: %s", audioURL)
			c.Header("X-Error", "audio URL is not allowed for outgoing requests")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		// download audio from url
		req, err := http.NewRequestWithContext(c.Request.Context(), "GET", audioURL, nil)
		if err != nil {
			s.Logw("transcribe", "failed to create request for audio URL: %v", err)
			c.Header("X-Error", "failed to create request for audio URL")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			s.Logw("transcribe", "failed to download audio from URL: %s: %v", audioURL, err)
			c.Header("X-Error", "failed to download audio from URL")
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}
		audioDataFile = resp.Body
		inputMIME = resp.Header.Get("Content-Type")
		if inputMIME != "" {
			inputMIME = strings.TrimSpace(strings.Split(inputMIME, ";")[0])
		}
	}
	defer audioDataFile.Close()

	// read audio data from file
	var streamer beep.Streamer
	var format beep.Format
	switch inputMIME {
	case "audio/wav":
		streamer, format, err = wav.Decode(audioDataFile)
	case "audio/mpeg":
		streamer, format, err = mp3.Decode(audioDataFile)
	case "audio/ogg":
		streamer, format, err = vorbis.Decode(audioDataFile)
	case "audio/flac":
		streamer, format, err = flac.Decode(audioDataFile)
	default:
		s.Logw("transcribe", "unsupported audio format: %s", inputMIME)
		c.Header("X-Error", "unsupported audio format")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}
	if err != nil {
		s.Logw("transcribe", "failed to decode audio data: %v", err)
		c.Header("X-Error", "failed to decode audio data")
		c.AbortWithStatus(http.StatusBadRequest)
		return
	}

	// do down mix
	streamer = &BeepDownMixer{
		source:        streamer,
		downMixMethod: DownMixMethod(downMixMethod),
	}

	channel := make(chan ASRResult)
	s.executorPool.Dispatch(func() {
		inferencer, err := s.memoryPool.GetObj(c.Request.Context(), modelName)
		if err != nil {
			s.Logw("matting", "failed to get inferencer: %v", err)
			close(channel)
			return
		}

		sampleRater, ok := inferencer.(SampleRater)
		if !ok {
			// what the fuck?
			s.Logw("transcribe", "inferencer does not implement SampleRater interface")
			c.Header("X-Error", "inferencer does not implement SampleRater interface")
			c.AbortWithStatus(http.StatusInternalServerError)
			close(channel)
			return
		}
		sampleRate := sampleRater.SampleRate()

		if format.SampleRate != beep.SampleRate(sampleRate) {
			// do SRC
			streamer = beep.Resample(4, format.SampleRate, beep.SampleRate(sampleRate), streamer)
		}

		// convert to []float32 for transcribe
		float32Samples := beepConvertSamplesToFloat32Array(streamer)

		// transcribe
		result, err := inferencer.Transcribe(c.Request.Context(), float32Samples, hotwords)
		if err != nil {
			s.Logw("transcribe", "failed to transcribe: %v", err)
			close(channel)
			return
		}

		channel <- result
		close(channel)
	})

	result, ok := <-channel
	if !ok {
		c.Header("X-Error", "failed to transcribe")
		c.AbortWithStatus(http.StatusInternalServerError)
		return
	}
	c.JSON(http.StatusOK, result)
}
