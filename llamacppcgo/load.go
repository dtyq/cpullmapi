package llamacppcgo

/*
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"
import (
	"unsafe"
)

var libraryLoaded bool = false

func LoadLibrary(libllamaPath, libmtmdPath string) error {
	if libraryLoaded {
		return nil
	}

	cLibllamaPath := C.CString(libllamaPath)
	defer C.free(unsafe.Pointer(cLibllamaPath))
	cLibmtmdPath := C.CString(libmtmdPath)
	defer C.free(unsafe.Pointer(cLibmtmdPath))

	if errCode := C.LCCLoadLibrary(cLibllamaPath, cLibmtmdPath); errCode != C.LCC_ERROR_SUCCESS {
		return ErrorCode(errCode)
	}
	libraryLoaded = true
	return nil
}
