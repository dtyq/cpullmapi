package llamacppcgo

import (
	"fmt"
	"testing"

	"github.com/dtyq/cpullmapi/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestSetLlamaLogFunc(t *testing.T) {
	testutil.RequireFiles(t, libllamaPath, libmtmdPath)

	var err error

	err = LoadLibrary(libllamaPath, libmtmdPath)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	err = SetLlamaLogFunc(func(level GGMLLogLevel, message string) {
		fmt.Printf("[LlamaLog][%d] %s", level, message)
	})
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	_, err = NewSession(SessionConfig{
		ModelPath: ".",
		Sampler: SamplerConfig{
			Kind: SamplerKindGreedy,
		},
	})
	if !assert.Error(t, err) {
		t.FailNow()
	}

	err = SetLlamaLogFunc(nil)
	if !assert.NoError(t, err) {
		t.FailNow()
	}

	_, err = NewSession(SessionConfig{
		ModelPath: ".",
		Sampler: SamplerConfig{
			Kind: SamplerKindGreedy,
		},
	})
	if !assert.Error(t, err) {
		t.FailNow()
	}

	SetLlamaLogFunc(nil)
}
