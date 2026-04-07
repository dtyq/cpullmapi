package cpullmapi

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/davidbyttow/govips/v2/vips"
	ort "github.com/yalue/onnxruntime_go"
)

type ONNXSODInferencer struct {
	session         *ort.AdvancedSession
	sessionMutex    sync.Mutex
	preprocessor    *ViTImageProcessor
	postprocessFunc imagePostprocessFunc[ONNXSODInferencer]
	inputTensors    []ort.Value
	outputTensors   []ort.Value

	width  int
	height int
}

func NewONNXSODInferencer(
	modelPath string,
	preprocessorConfigPath string,
	inputName string,
	outputName string,
	postprocessFunc imagePostprocessFunc[ONNXSODInferencer],
	sessionOptionsFunc ortSessionOptionsFunc,
) (*ONNXSODInferencer, error) {
	var err error
	preProcesserConfigJSON, err := os.ReadFile(preprocessorConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read preprocessor config: %v", err)
	}
	var preProcesserConfig ViTImageProcessorConfig
	err = json.Unmarshal(preProcesserConfigJSON, &preProcesserConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal image processor config: %v", err)
	}

	preprocessor, err := NewViTImageProcessor(preProcesserConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create image processor: %v", err)
	}

	width := preProcesserConfig.Size.Width
	height := preProcesserConfig.Size.Height
	inputTensors := make([]ort.Value, 1)
	inputTensors[0], err = ort.NewEmptyTensor[float32](ort.Shape{1, 3, int64(width), int64(height)})
	outputTensors := make([]ort.Value, 1)
	outputTensors[0], err = ort.NewEmptyTensor[float32](ort.Shape{1, 1, int64(width), int64(height)})

	opts, err := sessionOptionsFunc()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %v", err)
	}
	defer opts.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{inputName},
		[]string{outputName},
		inputTensors,
		outputTensors,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	return &ONNXSODInferencer{
		session:         session,
		preprocessor:    preprocessor,
		inputTensors:    inputTensors,
		outputTensors:   outputTensors,
		width:           width,
		height:          height,
		postprocessFunc: postprocessFunc,
	}, nil
}

func (in *ONNXSODInferencer) SegmentImage(image *vips.ImageRef) ([]*vips.ImageRef, error) {
	image_, err := image.Copy()
	if err != nil {
		return nil, fmt.Errorf("failed to copy image: %v", err)
	}
	chwArray, err := in.preprocessor.PreprocessImage(image_)
	if err != nil {
		return nil, fmt.Errorf("failed to preprocess image: %v", err)
	}

	// copy CHW array to input tensor
	inputTensor := in.inputTensors[0].(*ort.Tensor[float32])
	copy(inputTensor.GetData(), chwArray)

	in.sessionMutex.Lock()
	err = in.session.Run()
	in.sessionMutex.Unlock()
	if err != nil {
		return nil, fmt.Errorf("failed to run session: %v", err)
	}

	outputTensor := in.outputTensors[0].(*ort.Tensor[float32])
	outputArray := outputTensor.GetData()

	segments, err := in.postprocessFunc(in, outputArray)
	if err != nil {
		return nil, fmt.Errorf("failed to postprocess output array: %v", err)
	}

	// since we have only one segment, just inplace resize it to the original image size
	err = inplaceResizeImage(segments[0], image.Width(), image.Height(), PILResampleMethodBilinear)
	if err != nil {
		return nil, fmt.Errorf("failed to resize segment: %v", err)
	}

	return segments, nil
}

func (i *ONNXSODInferencer) Close() {
	i.session.Destroy() // best effort, no error checking
}

func (i *ONNXSODInferencer) GetCapabilities() []Capability {
	return []Capability{CapabilityImageSegmentation}
}
