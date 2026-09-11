//go:build with_audio

package main

import (
	_ "github.com/dtyq/cpullmapi/backends/crispasr"
	_ "github.com/dtyq/cpullmapi/backends/lcc"
	_ "github.com/dtyq/cpullmapi/backends/sherpa_onnx"
)
