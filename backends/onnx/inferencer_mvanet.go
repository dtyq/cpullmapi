package onnx

import (
	ort "github.com/yalue/onnxruntime_go"

	core "github.com/dtyq/cpullmapi"
)

func mvanetONNXSessionOptions() (*ort.SessionOptions, error) {
	opts, err := defaultONNXSessionOptions()
	if err != nil {
		return nil, err
	}
	err = opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableBasic)
	if err != nil {
		return nil, err
	}
	return opts, nil
}

func NewONNXMVANetInferencer(
	config ONNXSODCommonConfig,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		config.ModelPath,
		config.PreprocessorConfigPath,
		"pixel_values",
		"alphas",
		ben2PostprocessFunc,
		mvanetONNXSessionOptions,
	)
}

func init() {
	core.InferencerFactoryMap["ONNXMVANet"] = NewONNXMVANetInferencer
}
