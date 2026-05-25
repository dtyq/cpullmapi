package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	core "github.com/dtyq/cpullmapi"
)

// TODO: configurable
func defaultONNXSessionOptions() (*ort.SessionOptions, error) {
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
	// err = opts.AppendExecutionProviderOpenVINO(map[string]string{
	// 	"device_type": "CPU",
	// 	"load_config": `{
	// 		"CPU": {
	// 			"PERFORMANCE_HINT": "LATENCY",
	// 			"INFERENCE_NUM_THREADS": "1",
	// 			"NUM_STREAMS": "2",
	// 			"ENABLE_CPU_PINNING": "yes"
	// 		}
	// 	}`,
	// })
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to append execution provider openvino: %v", err)
	// }
	return opts, nil
}

type ortSessionOptionsFunc func() (*ort.SessionOptions, error)

func init() {
	core.ConfigInitFuncs = append(core.ConfigInitFuncs, func(c *core.Config) error {
		var err error

		// initialize ort environment
		if c.Inference.ONNXSharedLibraryPath != "" {
			ort.SetSharedLibraryPath(c.Inference.ONNXSharedLibraryPath)

			err = ort.InitializeEnvironment()
			if err != nil {
				return fmt.Errorf("failed to initialize ort environment: %v", err)
			}
		}

		return err
	})

	core.ConfigShutdownFuncs = append(core.ConfigShutdownFuncs, func(c *core.Config) {
		if c.Inference.ONNXSharedLibraryPath != "" {
			// shutdown ort environment
			ort.DestroyEnvironment()
		}
	})
}
