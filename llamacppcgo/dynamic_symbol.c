
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <dlfcn.h>

#ifdef _cplusplus
extern "C" {
#endif

#if defined(__x86_64__)
#define STUB(name) \
void *_##name = NULL; \
__asm__( \
    ".pushsection .text." #name ", \"ax\", @progbits\n\t" \
    ".globl " #name "\n\t" \
    ".balign 16\n\t" \
    ".type " #name ", @function\n\t" \
    #name ":\n\t" \
    "endbr64\n\t" \
    "movq _" #name "(%rip), %r11\n\t" \
    "jmp *%r11\n\t" \
    ".popsection\n\t" \
);
#elif defined(__aarch64__)
#define STUB(name) \
void *_##name = NULL; \
__asm__( \
    ".globl " #name "\n\t" \
    ".type " #name ", @function\n\t" \
    #name ":\n\t" \
    "adrp x16, _" #name "@PAGE\n\t" \
    "ldr x16, [x16, _" #name "@PAGEOFF]\n\t" \
    "br x16\n\t" \
);
#else
#error "Unsupported architecture"
#endif

STUB(llama_log_set)
STUB(llama_get_memory)
STUB(llama_memory_seq_rm)
STUB(llama_memory_seq_add)
STUB(llama_memory_seq_keep)
STUB(llama_memory_clear)
STUB(llama_batch_init)
STUB(llama_batch_free)
STUB(llama_n_batch)
STUB(llama_n_ctx)
STUB(llama_tokenize)
STUB(llama_detokenize)
STUB(llama_model_default_params)
STUB(llama_model_load_from_file)
STUB(llama_model_get_vocab)
STUB(llama_model_free)
STUB(llama_token_to_piece)
STUB(llama_vocab_is_eog)
STUB(llama_context_default_params)
STUB(llama_init_from_model)
STUB(llama_sampler_chain_default_params)
STUB(llama_sampler_chain_init)
STUB(llama_sampler_init_greedy)
STUB(llama_sampler_chain_add)
STUB(llama_sampler_init_top_k)
STUB(llama_sampler_init_top_p)
STUB(llama_sampler_init_temp)
STUB(llama_sampler_init_dist)
STUB(llama_sampler_sample)
STUB(llama_sampler_accept)
STUB(llama_sampler_free)
STUB(llama_decode)
STUB(llama_free)
STUB(mtmd_context_params_default)
STUB(mtmd_init_from_file)
STUB(mtmd_tokenize)
STUB(mtmd_bitmap_init_from_audio)
STUB(mtmd_bitmap_free)
STUB(mtmd_input_chunks_init)
STUB(mtmd_input_chunks_get)
STUB(mtmd_input_chunks_free)
STUB(mtmd_input_chunks_size)
STUB(mtmd_input_chunk_get_type)
STUB(mtmd_input_chunk_get_n_tokens)
STUB(mtmd_free)
STUB(mtmd_default_marker)
STUB(mtmd_helper_eval_chunks)
STUB(mtmd_helper_eval_chunk_single)

typedef int LCCErrorCode;

LCCErrorCode LCCLoadLibrary(const char *libllamaPath, const char *libmtmdPath)
{
    LCCErrorCode ret = 0 /* LCC_ERROR_SUCCESS */;
    void *libllamaHandle = NULL;
    void *libmtmdHandle = NULL;

    // load the libraries
#define LOAD_LIBRARY(name) do { \
        name##Handle = dlopen(name##Path, RTLD_LAZY); \
        if (!name##Handle) { \
            ret = -8 /* LCC_ERROR_FAILED_LOAD_LIBRARY */; \
            fprintf(stderr, "Failed to load library %s: %s\n", name##Path, dlerror()); \
            goto end; \
        } \
    } while (0)

    LOAD_LIBRARY(libllama);
    LOAD_LIBRARY(libmtmd);

    // load the symbols
#define LOAD_SYMBOL(sym, handleName) do { \
        _##sym = dlsym(handleName##Handle, #sym); \
        if (!_##sym) { \
            ret = -9 /* LCC_ERROR_FAILED_LOAD_SYMBOL */; \
            fprintf(stderr, "Failed to load symbol %s: %s\n", #sym, dlerror()); \
            goto end; \
        } \
    } while (0)

    LOAD_SYMBOL(llama_log_set, libllama);
    LOAD_SYMBOL(llama_get_memory, libllama);
    LOAD_SYMBOL(llama_memory_seq_rm, libllama);
    LOAD_SYMBOL(llama_memory_seq_add, libllama);
    LOAD_SYMBOL(llama_memory_seq_keep, libllama);
    LOAD_SYMBOL(llama_memory_clear, libllama);
    LOAD_SYMBOL(llama_batch_init, libllama);
    LOAD_SYMBOL(llama_batch_free, libllama);
    LOAD_SYMBOL(llama_n_batch, libllama);
    LOAD_SYMBOL(llama_n_ctx, libllama);
    LOAD_SYMBOL(llama_tokenize, libllama);
    LOAD_SYMBOL(llama_detokenize, libllama);
    LOAD_SYMBOL(llama_model_default_params, libllama);
    LOAD_SYMBOL(llama_model_load_from_file, libllama);
    LOAD_SYMBOL(llama_model_get_vocab, libllama);
    LOAD_SYMBOL(llama_model_free, libllama);
    LOAD_SYMBOL(llama_token_to_piece, libllama);
    LOAD_SYMBOL(llama_vocab_is_eog, libllama);
    LOAD_SYMBOL(llama_context_default_params, libllama);
    LOAD_SYMBOL(llama_init_from_model, libllama);
    LOAD_SYMBOL(llama_sampler_chain_default_params, libllama);
    LOAD_SYMBOL(llama_sampler_chain_init, libllama);
    LOAD_SYMBOL(llama_sampler_init_greedy, libllama);
    LOAD_SYMBOL(llama_sampler_chain_add, libllama);
    LOAD_SYMBOL(llama_sampler_init_top_k, libllama);
    LOAD_SYMBOL(llama_sampler_init_top_p, libllama);
    LOAD_SYMBOL(llama_sampler_init_temp, libllama);
    LOAD_SYMBOL(llama_sampler_init_dist, libllama);
    LOAD_SYMBOL(llama_sampler_sample, libllama);
    LOAD_SYMBOL(llama_sampler_accept, libllama);
    LOAD_SYMBOL(llama_sampler_free, libllama);
    LOAD_SYMBOL(llama_decode, libllama);
    LOAD_SYMBOL(llama_free, libllama);
    LOAD_SYMBOL(mtmd_context_params_default, libmtmd);
    LOAD_SYMBOL(mtmd_init_from_file, libmtmd);
    LOAD_SYMBOL(mtmd_tokenize, libmtmd);
    LOAD_SYMBOL(mtmd_bitmap_init_from_audio, libmtmd);
    LOAD_SYMBOL(mtmd_bitmap_free, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunks_init, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunks_get, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunks_free, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunks_size, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunk_get_type, libmtmd);
    LOAD_SYMBOL(mtmd_input_chunk_get_n_tokens, libmtmd);
    LOAD_SYMBOL(mtmd_free, libmtmd);
    LOAD_SYMBOL(mtmd_default_marker, libmtmd);
    LOAD_SYMBOL(mtmd_helper_eval_chunks, libmtmd);
    LOAD_SYMBOL(mtmd_helper_eval_chunk_single, libmtmd);

end:

#define UNLOAD_LIBRARY(name) do { \
        if (name##Handle) { \
            dlclose(name##Handle); \
            name##Handle = NULL; \
        } \
    } while (0)

    if (ret != 0) {
        UNLOAD_LIBRARY(libllama);
    }

    return ret;
}

#ifdef _cplusplus
}
#endif
