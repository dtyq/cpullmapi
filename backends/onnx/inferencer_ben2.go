package onnx

import (
	"fmt"
	"unsafe"

	"github.com/davidbyttow/govips/v2/vips"

	core "github.com/dtyq/cpullmapi"
)

func ben2PostprocessFunc(in *ONNXSODInferencer, outputArray []float32) ([]*vips.ImageRef, error) {
	// since we have only one channel, we can directly use the output array as HWC array

	hwcBytes := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(outputArray))), len(outputArray)*4)
	segment, err := vips.NewImageFromMemory(hwcBytes, in.width, in.height, 1, vips.BandFormatFloat, vips.InterpretationSRGB)
	if err != nil {
		return nil, fmt.Errorf("failed to create image from go image: %v", err)
	}

	return []*vips.ImageRef{segment}, nil
}

func NewONNXBEN2Inferencer(
	config ONNXSODCommonConfig,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		config.ModelPath,
		config.PreprocessorConfigPath,
		"pixel_values",
		"alphas",
		ben2PostprocessFunc,
		defaultONNXSessionOptions,
	)
}

func init() {
	core.InferencerFactoryMap["ONNXBEN2"] = NewONNXBEN2Inferencer
}
