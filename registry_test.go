//go:build with_image

package cpullmapi_test

// The config tests look factories up by model type, and the backends are what
// register them. Nothing else pulls a backend into this test binary.
import (
	_ "github.com/dtyq/cpullmapi/backends/onnx"
)
