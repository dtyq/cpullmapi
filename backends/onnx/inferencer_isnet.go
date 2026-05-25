package onnx

import core "github.com/dtyq/cpullmapi"

func NewONNXISNetInferencer(
	config ONNXSODCommonConfig,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		config.ModelPath,
		config.PreprocessorConfigPath,
		"input",
		"output",
		ben2PostprocessFunc,
		core.DefaultONNXSessionOptions,
	)
}

func init() {
	core.InferencerFactoryMap["ONNXISNet"] = NewONNXISNetInferencer
}
