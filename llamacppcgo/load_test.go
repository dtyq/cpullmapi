package llamacppcgo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TODO: use enviroment variables for library paths in tests
const libllamaPath = "../../llama.cpp/build/bin/libllama.so"
const libmtmdPath = "../../llama.cpp/build/bin/libmtmd.so"

func TestLoadLibrary(t *testing.T) {
	var err error
	// missing library
	err = LoadLibrary("nonexistent_libllama.so", "nonexistent_libmtmd.so")
	if !assert.Error(t, err) {
		t.FailNow()
	}

	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	// test idempotency
	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}
}
