//go:build with_image

package cpullmapi

import (
	"context"
	"encoding/binary"
	"math"
	"net/http"
	"sync"
	"testing"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var pngMagic = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

// fakeSegmentationInferencer 按固定 alpha 值造一个单波段分割结果。
type fakeSegmentationInferencer struct {
	alpha float32
	err   error

	mu    sync.Mutex
	calls int
	size  [2]int
}

func (i *fakeSegmentationInferencer) GetCapabilities() []Capability {
	return []Capability{CapabilityImageSegmentation}
}

func (i *fakeSegmentationInferencer) Close() {}

func (i *fakeSegmentationInferencer) SegmentImage(
	_ context.Context, image *vips.ImageRef,
) ([]*vips.ImageRef, error) {
	i.mu.Lock()
	i.calls++
	i.size = [2]int{image.Width(), image.Height()}
	alpha, err := i.alpha, i.err
	i.mu.Unlock()

	if err != nil {
		return nil, err
	}

	width, height := image.Width(), image.Height()
	pixels := make([]byte, width*height*4)
	for offset := 0; offset < len(pixels); offset += 4 {
		binary.LittleEndian.PutUint32(pixels[offset:], math.Float32bits(alpha))
	}

	segment, err := vips.NewImageFromMemory(
		pixels, width, height, 1, vips.BandFormatFloat, vips.InterpretationBW)
	if err != nil {
		return nil, err
	}
	return []*vips.ImageRef{segment}, nil
}

func (i *fakeSegmentationInferencer) lastSize() [2]int {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.size
}

// testPNG 造一张 sRGB 的 PNG。把每个像素写成不同的值，好看出输出有没有被改坏。
func testPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	requireVips(t)

	pixels := make([]byte, width*height*3)
	for n := range pixels {
		pixels[n] = uint8(n * 7)
	}

	image, err := vips.NewImageFromMemory(
		pixels, width, height, 3, vips.BandFormatUchar, vips.InterpretationSRGB)
	require.NoError(t, err)
	defer image.Close()

	data, _, err := image.ExportPng(nil)
	require.NoError(t, err)
	return data
}

func openPNG(t *testing.T, data []byte) *vips.ImageRef {
	t.Helper()
	requireVips(t)

	require.GreaterOrEqual(t, len(data), len(pngMagic), "response body is too short to be an image")
	image, err := vips.NewImageFromBuffer(data)
	require.NoError(t, err)
	t.Cleanup(image.Close)
	return image
}

// rgbaOf 读回一张 RGBA 图的原始像素。
func rgbaOf(t *testing.T, image *vips.ImageRef) []byte {
	t.Helper()

	require.Equal(t, 4, image.Bands())
	data, err := image.ToBytes()
	require.NoError(t, err)
	require.Len(t, data, image.Width()*image.Height()*4)
	return data
}

func mattingRequest(
	t *testing.T, fields map[string]string, image []byte, imageType string,
) *http.Request {
	t.Helper()
	return multipartUpload(t, "/api/v1/matting", "imageData", "a.png", image, imageType, fields)
}

func newMattingServer(t *testing.T, inferencer Inferencer) *testServer {
	t.Helper()

	inferencers := map[string]Inferencer{}
	if inferencer != nil {
		inferencers[testMattingModel] = inferencer
	}
	return newImageTestServer(t, inferencers)
}

func TestMattingRequiresToken(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	req := mattingRequest(t, map[string]string{"modelName": testMattingModel}, nil, "")
	req.Header.Del("Authorization")

	assert.Equal(t, http.StatusUnauthorized, serveRequest(ts, req).Code)
}

func TestMattingRejectsMissingModelName(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, nil, nil, ""))
	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "modelName is required", recorder.Header().Get("X-Error"))
}

func TestMattingRejectsBadBackgroundColor(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName":       testMattingModel,
		"backgroundColor": "not-a-color",
	}, testPNG(t, 4, 4), "image/png"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "failed to parse background color", recorder.Header().Get("X-Error"))
}

func TestMattingRejectsMissingImage(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{"modelName": testMattingModel}, nil, ""))
	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "imageData or imageURL is required", recorder.Header().Get("X-Error"))
}

func TestMattingRejectsBothImageSources(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
		"imageURL":  "http://example.com/a.png",
	}, testPNG(t, 4, 4), "image/png"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t,
		"imageData and imageURL cannot be set at the same time",
		recorder.Header().Get("X-Error"))
}

func TestMattingRejectsCorruptImage(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, []byte("definitely not an image"), "image/png"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "failed to open image from buffer", recorder.Header().Get("X-Error"))
}

func TestMattingRejectsNonSegmentationModel(t *testing.T) {
	// DummyInferencer 是完整的 Inferencer，但不支持图像分割。
	ts := newMattingServer(t, &DummyInferencer{})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, testPNG(t, 4, 4), "image/png"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model "+testMattingModel+" does not support image segmentation",
		recorder.Header().Get("X-Error"))
}

func TestMattingFailsWhenModelIsMissing(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": "nope",
	}, testPNG(t, 4, 4), "image/png"))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "model nope not found", recorder.Header().Get("X-Error"))
}

func TestMattingFailsWhenSegmentErrors(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1, err: assert.AnError})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, testPNG(t, 4, 4), "image/png"))

	assert.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.Equal(t, "failed to segment image", recorder.Header().Get("X-Error"))
}

func TestMattingRejectsDisallowedImageURL(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
		"imageURL":  "http://example.com/a.png",
	}, nil, ""))

	assert.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "image URL is not allowed for outgoing requests", recorder.Header().Get("X-Error"))
}

func TestMattingReturnsPNGByDefault(t *testing.T) {
	inferencer := &fakeSegmentationInferencer{alpha: 1}
	ts := newMattingServer(t, inferencer)

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, testPNG(t, testMattingWidth, testMattingHeight), "image/png"))

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "image/png", recorder.Header().Get("Content-Type"))
	assert.Equal(t, pngMagic, recorder.Body.Bytes()[:len(pngMagic)])
	assert.Equal(t, [2]int{testMattingWidth, testMattingHeight}, inferencer.lastSize(),
		"the inferencer must see the image at its original size")

	// 全 1 的 mask：输出应该带 alpha，且尺寸与原图一致。
	output := openPNG(t, recorder.Body.Bytes())
	assert.Equal(t, testMattingWidth, output.Width())
	assert.Equal(t, testMattingHeight, output.Height())
	assert.Equal(t, 4, output.Bands())
}

func TestMattingZeroMaskProducesTransparentImage(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 0})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, testPNG(t, testMattingWidth, testMattingHeight), "image/png"))

	require.Equal(t, http.StatusOK, recorder.Code)

	// 全 0 的 mask：整张图应该完全透明。
	pixels := rgbaOf(t, openPNG(t, recorder.Body.Bytes()))
	for offset := 3; offset < len(pixels); offset += 4 {
		require.Zero(t, pixels[offset], "a zero mask must zero out the alpha channel")
	}
}

func TestMattingHonoursAcceptHeader(t *testing.T) {
	tcs := []struct {
		accept   string
		wantType string
	}{
		{accept: "image/jpeg", wantType: "image/jpeg"},
		{accept: "image/png", wantType: "image/png"},
	}

	for _, tc := range tcs {
		t.Run(tc.accept, func(t *testing.T) {
			ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})

			req := mattingRequest(t, map[string]string{
				"modelName": testMattingModel,
			}, testPNG(t, testMattingWidth, testMattingHeight), "image/png")
			req.Header.Set("Accept", tc.accept)

			recorder := serveRequest(ts, req)
			require.Equal(t, http.StatusOK, recorder.Code)
			assert.Equal(t, tc.wantType, recorder.Header().Get("Content-Type"))
		})
	}
}

func TestMattingAppliesBackgroundColor(t *testing.T) {
	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 0})

	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName":       testMattingModel,
		"backgroundColor": "#ff0000ff",
	}, testPNG(t, testMattingWidth, testMattingHeight), "image/png"))

	require.Equal(t, http.StatusOK, recorder.Code)

	// 全透明的 mask 加上不透明的红底：整张图应该都是红的。
	pixels := rgbaOf(t, openPNG(t, recorder.Body.Bytes()))
	for offset := 0; offset < len(pixels); offset += 4 {
		if pixels[offset] != 255 || pixels[offset+1] != 0 || pixels[offset+2] != 0 || pixels[offset+3] != 255 {
			require.Fail(t, "pixel is not the background colour",
				"got rgba(%d,%d,%d,%d)",
				pixels[offset], pixels[offset+1], pixels[offset+2], pixels[offset+3])
		}
	}
}

func TestMattingAcceptsGrayscaleInput(t *testing.T) {
	requireVips(t)

	pixels := make([]byte, testMattingWidth*testMattingHeight)
	for n := range pixels {
		pixels[n] = uint8(n * 3)
	}
	gray, err := vips.NewImageFromMemory(
		pixels, testMattingWidth, testMattingHeight, 1, vips.BandFormatUchar, vips.InterpretationBW)
	require.NoError(t, err)
	defer gray.Close()

	grayPNG, _, err := gray.ExportPng(nil)
	require.NoError(t, err)

	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})
	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, grayPNG, "image/png"))

	require.Equal(t, http.StatusOK, recorder.Code)

	output := openPNG(t, recorder.Body.Bytes())
	assert.Equal(t, 4, output.Bands(), "grayscale is promoted to sRGB and gets an alpha channel")
}

func TestMattingAcceptsFourChannelInput(t *testing.T) {
	requireVips(t)

	pixels := make([]byte, testMattingWidth*testMattingHeight*4)
	for n := range pixels {
		pixels[n] = uint8(n * 5)
	}
	rgba, err := vips.NewImageFromMemory(
		pixels, testMattingWidth, testMattingHeight, 4, vips.BandFormatUchar, vips.InterpretationSRGB)
	require.NoError(t, err)
	defer rgba.Close()

	rgbaPNG, _, err := rgba.ExportPng(nil)
	require.NoError(t, err)

	ts := newMattingServer(t, &fakeSegmentationInferencer{alpha: 1})
	recorder := serveRequest(ts, mattingRequest(t, map[string]string{
		"modelName": testMattingModel,
	}, rgbaPNG, "image/png"))

	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, pngMagic, recorder.Body.Bytes()[:len(pngMagic)])
}
