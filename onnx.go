package cpullmapi

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// TODO: configurable
func onnxSessionOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %v", err)
	}
	if err := opts.SetIntraOpNumThreads(2); err != nil {
		return nil, fmt.Errorf("failed to set intra op num threads: %v", err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("failed to set inter op num threads: %v", err)
	}
	err = opts.AppendExecutionProviderOpenVINO(map[string]string{
		"device_type": "CPU",
		"load_config": `{
			"CPU": {
				"PERFORMANCE_HINT":        "LATENCY",
				"INFERENCE_NUM_THREADS":   "2",
				"NUM_STREAMS":             "1"
			}
		}`,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to append execution provider openvino: %v", err)
	}
	return opts, nil
}

type ortSessionOptionsFunc func() (*ort.SessionOptions, error)
