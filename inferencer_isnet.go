package cpullmapi

func NewONNXISNetInferencer(
	config ONNXSODCommonConfig,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		config.ModelPath,
		config.PreprocessorConfigPath,
		"input",
		"output",
		ben2PostprocessFunc,
		onnxSessionOptions,
	)
}

func init() {
	InferencerFactoryMap["ONNXISNet"] = NewONNXISNetInferencer
}
