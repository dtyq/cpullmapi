package llamacppcgo

/*
#include <stdint.h>
#include <stdlib.h>
#include "wrapper.h"

extern void lccGoPlannerRollbackDispatch(
    uintptr_t handle,
    uint64_t firstSequence
);
extern void lccGoPlannerBlockDispatch(
    uintptr_t handle,
    LCCStreamingASRChunkBlock *block
);

static void lccGoPlannerRollbackCallback(
    void *userData,
    uint64_t firstSequence
) {
    lccGoPlannerRollbackDispatch(
        (uintptr_t)userData,
        firstSequence
    );
}

static void lccGoPlannerBlockCallback(
    void *userData,
    const LCCStreamingASRChunkBlock *block
) {
    lccGoPlannerBlockDispatch(
        (uintptr_t)userData,
        (LCCStreamingASRChunkBlock *)block
    );
}

static LCCStreamingASRChunkCallbacks lccGoPlannerCallbacks(
    uintptr_t handle
) {
    return (LCCStreamingASRChunkCallbacks) {
        .userData = (void *)handle,
        .onRollback = lccGoPlannerRollbackCallback,
        .onBlock = lccGoPlannerBlockCallback,
    };
}
*/
import "C"
import (
	"fmt"
	"os"
	"regexp"
	"runtime/cgo"
	"unsafe"
)

// Qwen3ASRSession is the Go-facing session abstraction. C owns only the PCM
// planner. Go receives planner events and performs model/KV orchestration.
type Qwen3ASRSession struct {
	*Session
	currentPos          C.int32_t
	vocab               *C.struct_llama_vocab
	planner             *C.LCCStreamingASRChunkState
	outputTokens        *C.llama_token
	outputTokenCount    C.size_t
	provisionalAudioPos C.int32_t
	rTrimTokens         C.size_t
	callbackErr         error
	handle              cgo.Handle
	trace               bool
	initialized         bool
	systemPrompt        string
}

func NewQwen3ASRSession(session *Session) *Qwen3ASRSession {
	result := &Qwen3ASRSession{
		Session: session,
		vocab:   C.llama_model_get_vocab(session.lccCtx.model),
		trace:   os.Getenv("LCC_ASR_TRACE") != "",
	}
	result.handle = cgo.NewHandle(result)
	return result
}

func (s *Qwen3ASRSession) SetSystemPrompt(prompt string) error {
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	if s.initialized {
		C.LCCStreamingASRChunkStateFree(s.planner)
		s.planner = nil
		s.initialized = false
	}
	C.free(unsafe.Pointer(s.outputTokens))
	s.outputTokens = nil
	s.outputTokenCount = 0
	if errCode := C.LCCStreamingASRQwen3ASRSetPrompt(
		&s.lccCtx,
		&s.currentPos,
		cPrompt,
		C.size_t(len(prompt)),
	); errCode != C.LCC_ERROR_SUCCESS {
		return ErrorCode(errCode)
	}

	callbacks := C.lccGoPlannerCallbacks(C.uintptr_t(s.handle))
	s.planner = C.LCCStreamingASRChunkStateCreate(&callbacks)
	if s.planner == nil {
		return ErrorCode(C.LCC_ERROR_FAILED_ALLOCATE_MEMORY)
	}
	if s.trace {
		fmt.Fprintf(os.Stderr, "LCC ASR prompt pos=%d\n", int(s.currentPos))
	}
	s.outputTokens = nil
	s.outputTokenCount = 0
	s.provisionalAudioPos = s.currentPos
	s.rTrimTokens = 0
	s.callbackErr = nil
	s.systemPrompt = prompt
	s.initialized = true
	return nil
}

var qwen3asrResultRe = regexp.MustCompile(`^\s*([^<]+)<asr_text>(.+)$`)

func (s *Qwen3ASRSession) result() (lang string, text string, err error) {
	if s.outputTokenCount == 0 {
		return "", "", nil
	}

	nBytes := C.llama_detokenize(
		s.vocab,
		s.outputTokens,
		C.int32_t(s.outputTokenCount),
		nil,
		0,
		false,
		false,
	)
	if nBytes >= 0 {
		return "", "", fmt.Errorf("unexpected Qwen3-ASR detokenize size: %d", nBytes)
	}

	textBuf := make([]byte, -nBytes)
	if C.llama_detokenize(
		s.vocab,
		s.outputTokens,
		C.int32_t(s.outputTokenCount),
		(*C.char)(unsafe.Pointer(&textBuf[0])),
		C.int32_t(len(textBuf)),
		false,
		false,
	) < 0 {
		return "", "", fmt.Errorf("failed to detokenize Qwen3-ASR output")
	}

	matches := qwen3asrResultRe.FindSubmatch(textBuf)
	if len(matches) == 3 {
		return string(matches[1]), string(matches[2]), nil
	}
	// Early provisional rounds may only reach the language marker and
	// <asr_text> before enough audio exists to produce language/text.
	return "", string(textBuf), nil
}

func (s *Qwen3ASRSession) FeedAudioSamples(
	samples []float32,
	rTrimTokens uint,
) (lang string, text string, err error) {
	if !s.initialized {
		return "", "", fmt.Errorf("Qwen3-ASR system prompt is not set")
	}
	if len(samples) == 0 {
		return s.result()
	}

	s.callbackErr = nil
	s.rTrimTokens = C.size_t(rTrimTokens)
	if errCode := C.LCCStreamingASRChunkStateFeed(
		s.planner,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
	); errCode != C.LCC_ERROR_SUCCESS {
		return "", "", ErrorCode(errCode)
	}
	if s.callbackErr != nil {
		return "", "", s.callbackErr
	}
	return s.result()
}

// Flush replaces the last silence-padded provisional block with the real final
// tail. Call it once after the last audio callback to obtain final text.
func (s *Qwen3ASRSession) Flush(
	rTrimTokens uint,
) (lang string, text string, err error) {
	if !s.initialized {
		return "", "", fmt.Errorf("Qwen3-ASR system prompt is not set")
	}

	s.callbackErr = nil
	s.rTrimTokens = C.size_t(rTrimTokens)
	if errCode := C.LCCStreamingASRChunkStateFlush(s.planner); errCode != C.LCC_ERROR_SUCCESS {
		return "", "", ErrorCode(errCode)
	}
	if s.callbackErr != nil {
		return "", "", s.callbackErr
	}
	return s.result()
}

func (s *Qwen3ASRSession) Reset() error {
	if s.systemPrompt == "" {
		return fmt.Errorf("Qwen3-ASR system prompt is not set")
	}
	return s.SetSystemPrompt(s.systemPrompt)
}

func (s *Qwen3ASRSession) Close() {
	if s.initialized {
		if s.trace {
			fmt.Fprintln(os.Stderr, "LCC ASR close planner")
		}
		C.LCCStreamingASRChunkStateFree(s.planner)
		s.planner = nil
		s.initialized = false
	}
	if s.trace {
		fmt.Fprintln(os.Stderr, "LCC ASR close output")
	}
	C.free(unsafe.Pointer(s.outputTokens))
	s.outputTokens = nil
	s.outputTokenCount = 0
	if s.handle != 0 {
		s.handle.Delete()
		s.handle = 0
	}
	if s.trace {
		fmt.Fprintln(os.Stderr, "LCC ASR close context")
	}
	s.Session.Close()
	if s.trace {
		fmt.Fprintln(os.Stderr, "LCC ASR close done")
	}
}

//export lccGoPlannerRollbackDispatch
func lccGoPlannerRollbackDispatch(handle C.uintptr_t, firstSequence C.uint64_t) {
	session := cgo.Handle(handle).Value().(*Qwen3ASRSession)
	_ = firstSequence
	if !session.initialized || session.callbackErr != nil {
		return
	}
	if session.trace {
		fmt.Fprintf(
			os.Stderr,
			"LCC ASR rollback sequence=%d audioPos=%d..%d\n",
			uint64(firstSequence),
			int(session.provisionalAudioPos),
			int(session.currentPos),
		)
	}
	if !C.llama_memory_seq_rm(
		C.llama_get_memory(session.lccCtx.ctx),
		0,
		C.llama_pos(session.provisionalAudioPos),
		C.llama_pos(session.currentPos),
	) {
		session.callbackErr = fmt.Errorf("failed to rollback provisional audio KV")
		return
	}
	session.currentPos = session.provisionalAudioPos
}

//export lccGoPlannerBlockDispatch
func lccGoPlannerBlockDispatch(
	handle C.uintptr_t,
	block *C.LCCStreamingASRChunkBlock,
) {
	session := cgo.Handle(handle).Value().(*Qwen3ASRSession)
	if !session.initialized || session.callbackErr != nil || block == nil {
		return
	}
	if block.flags&C.LCC_STREAMING_ASR_BLOCK_PROVISIONAL != 0 {
		session.provisionalAudioPos = session.currentPos
	}
	if session.trace {
		fmt.Fprintf(
			os.Stderr,
			"LCC ASR block sequence=%d flags=0x%x samples=%d..%d nSamples=%d audioPos=%d\n",
			uint64(block.sequence),
			uint(block.flags),
			uint64(block.sampleStart),
			uint64(block.sampleEnd),
			uint64(block.nSamples),
			int(session.currentPos),
		)
	}

	errCode := C.LCCStreamingASRQwen3ASRFeedSamples(
		&session.lccCtx,
		&session.currentPos,
		block.pcmData,
		block.nSamples,
		&session.outputTokens,
		&session.outputTokenCount,
		session.rTrimTokens,
	)
	if errCode != C.LCC_ERROR_SUCCESS {
		session.callbackErr = ErrorCode(errCode)
	}
}
