
#ifndef _LCC_WRAPPER_H
#define _LCC_WRAPPER_H
// LCC for 'L'lama'C'pp'C'go

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _LCCErrorCode {
    LCC_ERROR_SUCCESS = 0, // "success"
    LCC_ERROR_NOT_IMPLEMENTED = -1, // "not implemented"
    LCC_ERROR_UNKNOWN = -2, // "unknown error"
    LCC_ERROR_FAILED_ALLOCATE_MEMORY = -3, // "failed to allocate memory"
    LCC_ERROR_INVALID_ARGUMENT = -4, // "invalid argument"
    LCC_ERROR_FAILED_LOAD_MODEL = -5, // "failed to load model"
    LCC_ERROR_FAILED_CREATE_CONTEXT = -6, // "failed to create llama.cpp context"
    LCC_ERROR_FAILED_CREATE_MTMD_CONTEXT = -7, // "failed to create MTMD context"
    LCC_ERROR_FAILED_CREATE_SAMPLER = -8, // "failed to create sampler"
    LCC_ERROR_FAILED_LOAD_LIBRARY = -9, // "failed to load library"
    LCC_ERROR_FAILED_LOAD_SYMBOL = -10, // "failed to load symbol"
    LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS = -11, // "failed to create MTMD input chunks"
    LCC_ERROR_FAILED_MTMD_TOKENIZE = -12, // "failed to tokenize input with MTMD"
    LCC_ERROR_FAILED_MTMD_EVAL = -13, // "failed to eval input with MTMD"
    LCC_ERROR_FAILED_SAMPLE = -14, // "failed to sample token"
    LCC_ERROR_FAILED_CREATE_AUDIO_BITMAP = -15, // "failed to create MTMD audio bitmap"
    LCC_ERROR_FAILED_LLM_DECODE = -16, // "failed to do LLM decode"
    LCC_ERROR_FAILED_TOKENIZE = -17, // "failed to tokenize text"
    LCC_ERROR_FAILED_DETOKENIZE = -18 // "failed to detokenize tokens"
} LCCErrorCode;

typedef enum _LCCSamplerKind {
    LCC_SAMPLER_GREEDY = 0,
    LCC_SAMPLER_DIST = 1,
} LCCSamplerKind;

typedef struct _LCCSamplerDistConfig {
    float temperature;
    float top_p;
    int32_t top_k;
    // float repeat_penalty;
    uint32_t seed;
} LCCSamplerDistConfig;

typedef struct _LCCSamplerConfig {
    LCCSamplerKind kind;
    union {
        LCCSamplerDistConfig dist;
    } config;
} LCCSamplerConfig;

typedef struct _LCCContextInitParams {
    const char *modelPath;
    const char *mmprojPath;
    struct llama_model_params modelParams;
    struct llama_context_params ctxParams;
    LCCSamplerConfig samplerConfig;
} LCCContextInitParams;

typedef struct _LCCContext {
    struct llama_model *model;
    struct llama_context *ctx;
    struct llama_sampler *sampler;
    // struct llama_batch batch;
    mtmd_context *mtmdContext;
} LCCContext;

// dynamic_symbol.c
LCCErrorCode LCCLoadLibrary(const char *libllamaPath, const char *libmtmdPath);
// context.c
void LCCDefaultContextInitParams(LCCContextInitParams *initParams);
LCCErrorCode LCCInitContext(LCCContext *lccCtx, const LCCContextInitParams *initParams);
void LCCFreeContext(LCCContext *lccCtx);
// streaming_asr.c
LCCErrorCode LCCStreamingASRInit(
    LCCContext *lccCtx,
    const char *initialPrompt,
    int32_t *pCurrentPos
);
LCCErrorCode LCCStreamingASRFeedSamples(
    LCCContext *lccCtx,
    int32_t *pCurrentPos,
    const float *pcmData,
    size_t nSamples,
    const char *generationPrompt,
    char **callerFreeOutputBuffer
);
LCCErrorCode LCCStreamingASRFeedSamplesWithTokens(
    LCCContext *lccCtx,
    int32_t *pCurrentPos,
    const float *pcmData,
    size_t nSamples,
    llama_token *generationPrompt,
    size_t generationPromptTokenCount,
    llama_token **callerFreeOutputBuffer,
    size_t *pOutputTokenCount
);
LCCErrorCode LCCStreamingASRQwen3ASRSetPrompt(
    LCCContext *lccCtx,
    int32_t *pNewPos,
    const char *newPrompt,
    size_t sizeNewPrompt
);
LCCErrorCode LCCStreamingASRQwen3ASRFeedSamples(
    LCCContext *lccCtx,
    int32_t *pCurrentPos,
    const float *pcmData,
    size_t nSamples,
    llama_token **calleeAllocateCallerFreeOutputTokens,
    size_t *pOutputTokenCount,
    size_t rTrimTokens
);

enum {
    LCC_STREAMING_ASR_BLOCK_STABLE      = 1u << 0,
    LCC_STREAMING_ASR_BLOCK_PROVISIONAL = 1u << 1,
    LCC_STREAMING_ASR_BLOCK_FINAL       = 1u << 2,
};

typedef struct _LCCStreamingASRChunkBlock {
    uint64_t sequence;
    const float *pcmData;       // Borrowed; valid until the next state call.
    size_t nSamples;            // Padded to 15840 for provisional blocks.
    size_t nEffectiveFrames;   // Real mel frames before model padding.
    size_t nPaddedFrames;      // Qwen3A projector input frame count.
    size_t sampleStart;        // Offset in the accumulated PCM.
    size_t sampleEnd;          // Exclusive offset of real PCM.
    size_t nAudioTokens;       // 13 for each Qwen3A 100-frame block.
    unsigned int flags;         // LCC_STREAMING_ASR_BLOCK_* bitfield.
} LCCStreamingASRChunkBlock;

typedef struct _LCCStreamingASRChunkCallbacks {
    void *userData;
    void (*onRollback)(void *userData, uint64_t firstSequence);
    void (*onBlock)(void *userData, const LCCStreamingASRChunkBlock *block);
} LCCStreamingASRChunkCallbacks;

enum {
    LCC_STREAMING_ASR_CHUNK_HAS_PROVISIONAL = 1u << 0,
    LCC_STREAMING_ASR_CHUNK_FLUSHED = 1u << 1,
};

typedef struct _LCCStreamingASRChunkState {
    LCCStreamingASRChunkCallbacks callbacks;
    float *pcm;
    size_t pcmCount;
    size_t pcmCapacity;
    float *provisionalPcm;
    size_t provisionalCapacity;
    uint64_t nextSequence;
    uint64_t provisionalFirstSequence;
    size_t emittedStableBlocks;
    unsigned int flags;
} LCCStreamingASRChunkState;

LCCStreamingASRChunkState *LCCStreamingASRChunkStateCreate(
    const LCCStreamingASRChunkCallbacks *callbacks);
void LCCStreamingASRChunkStateFree(LCCStreamingASRChunkState *state);
LCCErrorCode LCCStreamingASRChunkStateFeed(
    LCCStreamingASRChunkState *state,
    const float *pcmData,
    size_t nSamples);
LCCErrorCode LCCStreamingASRChunkStateFlush(LCCStreamingASRChunkState *state);
void LCCStreamingASRChunkStateReset(LCCStreamingASRChunkState *state);

#ifdef __cplusplus
}
#endif

#endif // _LCC_WRAPPER_H
