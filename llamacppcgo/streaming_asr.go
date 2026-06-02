package llamacppcgo

/*
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"
import (
	"fmt"
	"regexp"
	"unsafe"
)

type Qwen3ASRSession struct {
	*Session
	currentPos    C.int32_t
	tokens        []C.llama_token
	vocab         *C.struct_llama_vocab
	outputTokens  *C.llama_token
	nOutputTokens C.size_t
}

func NewQwen3ASRSession(session *Session) *Qwen3ASRSession {
	return &Qwen3ASRSession{
		Session:    session,
		currentPos: 0,
		tokens:     make([]C.llama_token, 0),
		vocab:      C.llama_model_get_vocab(session.lccCtx.model),
	}
}

func (s *Qwen3ASRSession) SetSystemPrompt(prompt string) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	if errCode := C.LCCStreamingASRQwen3ASRSetPrompt(
		&s.lccCtx,
		&s.currentPos,
		cPrompt,
		C.size_t(len(prompt)),
	); errCode != C.LCC_ERROR_SUCCESS {
		return ErrorCode(errCode)
	}
	return nil
}

var qwen3asrResultRe *regexp.Regexp = regexp.MustCompile(`^\s*([^<]+)<asr_text>(.+)$`)

func (s *Qwen3ASRSession) FeedAudioSamples(samples []float32, rTrimTokens uint) (lang string, text string, err error) {
	if errCode := C.LCCStreamingASRQwen3ASRFeedSamples(
		&s.lccCtx,
		&s.currentPos,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
		&s.outputTokens,
		&s.nOutputTokens,
		C.size_t(rTrimTokens),
	); errCode != C.LCC_ERROR_SUCCESS {
		return "", "", ErrorCode(errCode)
	}

	// get length
	nTokens := C.llama_detokenize(s.vocab, s.outputTokens, C.int32_t(s.nOutputTokens), nil, 0, false, false)
	if nTokens == 0 {
		// no tokens generated
		return "", "", nil
	}
	textBuf := make([]byte, -nTokens)
	C.llama_detokenize(s.vocab, s.outputTokens, C.int32_t(s.nOutputTokens), (*C.char)(unsafe.Pointer(&textBuf[0])), C.int32_t(len(textBuf)), false, false)

	matches := qwen3asrResultRe.FindSubmatch(textBuf)
	if len(matches) == 3 {
		return string(matches[1]), string(matches[2]), nil
	} else {
		// unexpected format, return the whole text as is
		return "", "", fmt.Errorf("unexpected ASR output format: %s", string(textBuf))
	}
}

func (s *Qwen3ASRSession) Reset() {}

func (s *Qwen3ASRSession) Close() {
	C.free(unsafe.Pointer(s.outputTokens))
	s.Session.Close()
}
