package onnx

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/davidbyttow/govips/v2/vips"

	core "github.com/dtyq/cpullmapi"
)

func birefnetPostprocessFunc(in *ONNXSODInferencer, outputArray []float32) ([]*vips.ImageRef, error) {
	// since we have only one channel, we can directly use the output array as HWC array

	var x float64
	for i := range len(outputArray) {
		// sigmoid
		x = float64(outputArray[i])
		outputArray[i] = float32(1 / (1 + math.Exp(-x)))
	}
	hwcBytes := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(outputArray))), len(outputArray)*4)
	segment, err := vips.NewImageFromHWCArray(hwcBytes, 1, in.width, in.height, vips.BandFormatFloat, vips.InterpretationSRGB)
	if err != nil {
		return nil, fmt.Errorf("failed to create image from array: %v", err)
	}

	return []*vips.ImageRef{segment}, nil
}

func NewONNXBiRefNetInferencer(
	config ONNXSODCommonConfig,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		config.ModelPath,
		config.PreprocessorConfigPath,
		"input_image",
		"output_image",
		birefnetPostprocessFunc,
		core.DefaultONNXSessionOptions,
	)
}

func init() {
	core.InferencerFactoryMap["ONNXBiRefNet"] = NewONNXBiRefNetInferencer
}
