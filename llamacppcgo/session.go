package llamacppcgo

/*
#include <stdlib.h>
#include <string.h>
#include "wrapper.h"
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

type SamplerKind string

const (
	SamplerKindDist   SamplerKind = "dist"
	SamplerKindGreedy SamplerKind = "greedy"
)

type SamplerConfig struct {
	Kind        SamplerKind `json:"kind" yaml:"kind"` // "dist" or "greedy"
	Temperature float64     `json:"temperature" yaml:"temperature"`
	TopP        float64     `json:"topP" yaml:"topP"`
	TopK        int         `json:"topK" yaml:"topK"`
	// RepeatPenalty float64     `json:"repeatPenalty" yaml:"repeatPenalty"` // TODO
	Seed int64 `json:"seed" yaml:"seed"`
}

type SessionConfig struct {
	ModelPath    string        `json:"modelPath" yaml:"modelPath"`
	MMProjPath   string        `json:"mmprojPath" yaml:"mmprojPath"`
	NumGPULayers *int32        `json:"numGPULayers" yaml:"numGPULayers"`
	UseMmap      *bool         `json:"useMmap" yaml:"useMmap"`
	UseMlock     *bool         `json:"useMlock" yaml:"useMlock"`
	NCtx         *uint32       `json:"nCtx" yaml:"nCtx"`
	NBatch       *uint32       `json:"nBatch" yaml:"nBatch"`
	NThreads     *int32        `json:"nThreads" yaml:"nThreads"`
	Sampler      SamplerConfig `json:"sampler" yaml:"sampler"`
}

type Session struct {
	lccCtx C.LCCContext
	mu     sync.Mutex
}

func NewSession(config SessionConfig) (*Session, error) {
	if !libraryLoaded {
		return nil, ErrLibraryNotLoaded
	}

	ret := &Session{}

	initParams := C.LCCContextInitParams{}
	C.LCCDefaultContextInitParams(&initParams)
	// override default params with config
	initParams.modelPath = C.CString(config.ModelPath)
	defer C.free(unsafe.Pointer(initParams.modelPath))
	initParams.mmprojPath = C.CString(config.MMProjPath)
	defer C.free(unsafe.Pointer(initParams.mmprojPath))
	if config.NumGPULayers != nil {
		initParams.modelParams.n_gpu_layers = C.int32_t(*config.NumGPULayers)
	}
	if config.UseMmap != nil {
		initParams.modelParams.use_mmap = C.bool(*config.UseMmap)
	}
	if config.UseMlock != nil {
		initParams.modelParams.use_mlock = C.bool(*config.UseMlock)
	}
	if config.NCtx != nil {
		initParams.ctxParams.n_ctx = C.uint32_t(*config.NCtx)
	}
	if config.NBatch != nil {
		initParams.ctxParams.n_batch = C.uint32_t(*config.NBatch)
	}
	if config.NThreads != nil {
		initParams.ctxParams.n_threads = C.int32_t(*config.NThreads)
		initParams.ctxParams.n_threads_batch = C.int32_t(*config.NThreads)
	}
	switch config.Sampler.Kind {
	case SamplerKindGreedy:
		initParams.samplerConfig.kind = C.LCC_SAMPLER_GREEDY
	case SamplerKindDist:
		initParams.samplerConfig.kind = C.LCC_SAMPLER_DIST
		distConfig := C.LCCSamplerDistConfig{
			temperature: C.float(config.Sampler.Temperature),
			top_p:       C.float(config.Sampler.TopP),
			top_k:       C.int32_t(config.Sampler.TopK),
			seed:        C.uint32_t(config.Sampler.Seed),
		}
		C.memcpy(
			unsafe.Pointer(&initParams.samplerConfig.config),
			unsafe.Pointer(&distConfig),
			C.sizeof_LCCSamplerDistConfig,
		)
	default:
		return nil, fmt.Errorf("unsupported sampler kind: %s", config.Sampler.Kind)
	}

	if errCode := C.LCCInitContext(&ret.lccCtx, &initParams); errCode != C.LCC_ERROR_SUCCESS {
		return nil, ErrorCode(errCode)
	}

	return ret, nil
}

func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	C.LCCFreeContext(&s.lccCtx)
}
