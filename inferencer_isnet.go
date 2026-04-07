package cpullmapi

func NewISNetInferencer(
	modelPath string,
	preprocessorConfigPath string,
) (*ONNXSODInferencer, error) {
	return NewONNXSODInferencer(
		modelPath,
		preprocessorConfigPath,
		"input",
		"output",
		ben2PostprocessFunc,
		onnxSessionOptions,
	)
}
