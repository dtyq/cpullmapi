
#ifndef _LCC_WRAPPER_H
#define _LCC_WRAPPER_H
// LCC for 'L'lama'C'pp'C'go

// Rename the dlopen-backed entry points to lcc_stub_*. These resolve to stubs
// emitted by dynamic_symbol.c, which would otherwise collide with the llama.cpp
// CrispASR links into the same binary. Placed before llama.h so the declarations
// get renamed along with every call site. dynamic_symbol.c defines the symbols
// and stringifies the real names for dlsym, and does not include this header.
#define llama_log_set lcc_stub_llama_log_set
#define llama_backend_init lcc_stub_llama_backend_init
#define llama_get_memory lcc_stub_llama_get_memory
#define llama_memory_seq_rm lcc_stub_llama_memory_seq_rm
#define llama_memory_seq_add lcc_stub_llama_memory_seq_add
#define llama_memory_seq_keep lcc_stub_llama_memory_seq_keep
#define llama_memory_clear lcc_stub_llama_memory_clear
#define llama_batch_init lcc_stub_llama_batch_init
#define llama_batch_free lcc_stub_llama_batch_free
#define llama_n_batch lcc_stub_llama_n_batch
#define llama_n_ctx lcc_stub_llama_n_ctx
#define llama_tokenize lcc_stub_llama_tokenize
#define llama_detokenize lcc_stub_llama_detokenize
#define llama_model_default_params lcc_stub_llama_model_default_params
#define llama_model_load_from_file lcc_stub_llama_model_load_from_file
#define llama_model_get_vocab lcc_stub_llama_model_get_vocab
#define llama_model_free lcc_stub_llama_model_free
#define llama_token_to_piece lcc_stub_llama_token_to_piece
#define llama_vocab_is_eog lcc_stub_llama_vocab_is_eog
#define llama_context_default_params lcc_stub_llama_context_default_params
#define llama_init_from_model lcc_stub_llama_init_from_model
#define llama_sampler_chain_default_params lcc_stub_llama_sampler_chain_default_params
#define llama_sampler_chain_init lcc_stub_llama_sampler_chain_init
#define llama_sampler_init_greedy lcc_stub_llama_sampler_init_greedy
#define llama_sampler_chain_add lcc_stub_llama_sampler_chain_add
#define llama_sampler_init_top_k lcc_stub_llama_sampler_init_top_k
#define llama_sampler_init_top_p lcc_stub_llama_sampler_init_top_p
#define llama_sampler_init_temp lcc_stub_llama_sampler_init_temp
#define llama_sampler_init_dist lcc_stub_llama_sampler_init_dist
#define llama_sampler_sample lcc_stub_llama_sampler_sample
#define llama_sampler_accept lcc_stub_llama_sampler_accept
#define llama_sampler_free lcc_stub_llama_sampler_free
#define llama_decode lcc_stub_llama_decode
#define llama_free lcc_stub_llama_free
#define mtmd_log_set lcc_stub_mtmd_log_set
#define mtmd_context_params_default lcc_stub_mtmd_context_params_default
#define mtmd_init_from_file lcc_stub_mtmd_init_from_file
#define mtmd_tokenize lcc_stub_mtmd_tokenize
#define mtmd_bitmap_init_from_audio lcc_stub_mtmd_bitmap_init_from_audio
#define mtmd_bitmap_free lcc_stub_mtmd_bitmap_free
#define mtmd_input_chunks_init lcc_stub_mtmd_input_chunks_init
#define mtmd_input_chunks_get lcc_stub_mtmd_input_chunks_get
#define mtmd_input_chunks_free lcc_stub_mtmd_input_chunks_free
#define mtmd_input_chunks_size lcc_stub_mtmd_input_chunks_size
#define mtmd_input_chunk_get_type lcc_stub_mtmd_input_chunk_get_type
#define mtmd_input_chunk_get_n_tokens lcc_stub_mtmd_input_chunk_get_n_tokens
#define mtmd_free lcc_stub_mtmd_free
#define mtmd_default_marker lcc_stub_mtmd_default_marker
#define mtmd_helper_log_set lcc_stub_mtmd_helper_log_set
#define mtmd_helper_eval_chunks lcc_stub_mtmd_helper_eval_chunks
#define mtmd_helper_eval_chunk_single lcc_stub_mtmd_helper_eval_chunk_single

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
