package llamacppcgo

import (
	"testing"

	"github.com/dtyq/cpullmapi/internal/testutil"
	"github.com/stretchr/testify/assert"
)

const libllamaPath = "../third_party/llama.cpp/build_go/bin/libllama.so"
const libmtmdPath = "../third_party/llama.cpp/build_go/bin/libmtmd.so"

func TestLoadLibrary(t *testing.T) {
	testutil.RequireFiles(t, libllamaPath, libmtmdPath)

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
