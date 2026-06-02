package llamacppcgo

/*
#include <stdlib.h>
#include "wrapper.h"

extern void lccLogCallbackGateway(int level, char *text, void *user_data);
*/
import "C"
import (
	"fmt"
	"os"
	"unsafe"
)

var ErrLibraryNotLoaded = fmt.Errorf("library is not loaded")

type GGMLLogLevel int

const (
	GGML_LOG_LEVEL_NONE  GGMLLogLevel = C.GGML_LOG_LEVEL_NONE
	GGML_LOG_LEVEL_DEBUG GGMLLogLevel = C.GGML_LOG_LEVEL_DEBUG
	GGML_LOG_LEVEL_INFO  GGMLLogLevel = C.GGML_LOG_LEVEL_INFO
	GGML_LOG_LEVEL_WARN  GGMLLogLevel = C.GGML_LOG_LEVEL_WARN
	GGML_LOG_LEVEL_ERROR GGMLLogLevel = C.GGML_LOG_LEVEL_ERROR
	GGML_LOG_LEVEL_CONT  GGMLLogLevel = C.GGML_LOG_LEVEL_CONT // continue previous log
)

//typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)

var logCallback func(level GGMLLogLevel, text string)
var gatewaySet bool = false

func SetLlamaLogFunc(fn func(level GGMLLogLevel, text string)) error {
	if !libraryLoaded {
		return ErrLibraryNotLoaded
	}
	if !gatewaySet {
		C.llama_log_set((C.ggml_log_callback)(unsafe.Pointer(C.lccLogCallbackGateway)), nil)
		gatewaySet = true
	}
	logCallback = fn
	return nil
}

//export lccLogCallbackGateway
func lccLogCallbackGateway(level C.int, cText *C.char, userData unsafe.Pointer) {
	text := C.GoString(cText)
	if logCallback != nil {
		logCallback(GGMLLogLevel(level), text)
	} else {
		// default log behavior in llama_log_callback_default: print to stderr
		os.Stderr.WriteString(text)
		os.Stderr.Sync()
	}
}
