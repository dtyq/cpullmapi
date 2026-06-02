
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>

#include "wrapper.h"

#ifdef _LCC_DEBUG
#define metric_start(msg) \
    do { \
        const char *hint = msg; \
        const char *startFile = __FILE__; \
        int startLine = __LINE__; \
        struct timeval start, end; \
        gettimeofday(&start, NULL);

#define metric_end() \
        gettimeofday(&end, NULL); \
        double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0; \
        fprintf(stderr, "%s:%d: %s: %0.1fms\n", startFile, startLine, hint, elapsed); \
    } while (0)
#else
#define metric_start(msg)
#define metric_end()
#endif

static inline LCCErrorCode tokenize(
    mtmd_context *ctx,
    const char *prompt,
    const float *pcmSamples,
    size_t nSamples,
    mtmd_input_chunks *outChunks
) {
    mtmd_bitmap *audioBitmaps = NULL;
    size_t nAudioBitmaps = 0;

    struct mtmd_input_text inputText = {
        .text = prompt,
        .add_special = true,
        .parse_special = true,
    };

    if (pcmSamples && nSamples > 0) {
        audioBitmaps = mtmd_bitmap_init_from_audio(nSamples, pcmSamples);
        if (!audioBitmaps) {
            return LCC_ERROR_FAILED_CREATE_AUDIO_BITMAP;
        }
        nAudioBitmaps = 1;
    }

    int32_t r = mtmd_tokenize(
        ctx,
        outChunks,
        &inputText,
        (const mtmd_bitmap**)&audioBitmaps, nAudioBitmaps
    );
    if (audioBitmaps) {
        mtmd_bitmap_free(audioBitmaps);
    }
    if (r != 0) {
        return LCC_ERROR_FAILED_MTMD_TOKENIZE;
    }

    return LCC_ERROR_SUCCESS;
}

static LCCErrorCode decodeAsBatch(
    LCCContext *lccCtx,
    const llama_token *tokens,
    size_t nTokens,
    int32_t *pCurrentPos,
    bool logitLast
) {
    int32_t r;
    uint32_t nBatch = llama_n_batch(lccCtx->ctx);
    llama_batch batch = llama_batch_init(nBatch, 0, 1);

    for (int32_t i = 0; i < nTokens;) {
        // split into batches
        batch.n_tokens = 0;
        for (; i < nTokens && batch.n_tokens < nBatch; i++) {
            int32_t j = batch.n_tokens;
            batch.token   [j]    = tokens[i];
            batch.pos     [j]    = (*pCurrentPos) + j;
            batch.n_seq_id[j]    = 1;
            batch.seq_id  [j][0] = 0;
            batch.logits  [j]    = false;

            batch.n_tokens++;
        }
        if (logitLast && i == nTokens) {
            batch.logits[batch.n_tokens - 1] = true;
        }
        r = llama_decode(lccCtx->ctx, batch);
        if (r != 0) {
            llama_batch_free(batch);
            return  LCC_ERROR_FAILED_LLM_DECODE;
        }
        *pCurrentPos += batch.n_tokens;
    }
    llama_batch_free(batch);
    return LCC_ERROR_SUCCESS;
}

LCCErrorCode LCCStreamingASRInit(
    LCCContext *lccCtx,
    const char *initialPrompt,
    int32_t *pCurrentPos
) {
    LCCErrorCode ret = LCC_ERROR_SUCCESS;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS;
    }

    ret = tokenize(lccCtx->mtmdContext, initialPrompt, NULL, 0, chunks);
    if (ret != LCC_ERROR_SUCCESS) {
        mtmd_input_chunks_free(chunks);
        return ret;
    }

    int32_t r = mtmd_helper_eval_chunks(
        lccCtx->mtmdContext,
        lccCtx->ctx,
        chunks,
        *pCurrentPos,
        0,
        llama_n_batch(lccCtx->ctx),
        true,
        pCurrentPos
    );
    mtmd_input_chunks_free(chunks);
    if (r != 0) {
        return LCC_ERROR_FAILED_MTMD_EVAL;
    }

    return LCC_ERROR_SUCCESS;
}


LCCErrorCode LCCStreamingASRFeedSamples(
    LCCContext *lccCtx,
    int32_t *pCurrentPos,
    const float *pcmData,
    size_t nSamples,
    const char *generationPrompt,
    char **callerFreeOutputBuffer
) {
    int32_t r;
    LCCErrorCode ret = LCC_ERROR_SUCCESS;
    int32_t endPos;
    const struct llama_vocab *vocab = llama_model_get_vocab(lccCtx->model);
    mtmd_input_chunks *chunks;
    llama_memory_t mem = llama_get_memory(lccCtx->ctx);
    llama_batch batch = llama_batch_init(1, 0, 1);

    chunks = mtmd_input_chunks_init();
    if (!chunks) {
        ret = LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS;
        goto end;
    }

    // do mtmd tokenization for audio tokens
    ret = tokenize(lccCtx->mtmdContext, mtmd_default_marker(), pcmData, nSamples, chunks);
    if (ret != LCC_ERROR_SUCCESS) {
        goto end;
    }

    // eval new audio chunk
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            continue;
        }

        metric_start("== EVAL NEW AUDIO CHUNK");
        r = mtmd_helper_eval_chunk_single(
            lccCtx->mtmdContext,
            lccCtx->ctx,
            chunk,
            *pCurrentPos,
            0,
            llama_n_batch(lccCtx->ctx),
            true,
            pCurrentPos
        );
        metric_end();
        if (r != 0) {
            ret = LCC_ERROR_FAILED_MTMD_EVAL;
            goto end;
        }
    }
    mtmd_input_chunks_free(chunks);
    chunks = mtmd_input_chunks_init(); // re-init for generation prompt
    if (!chunks) {
        ret = LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS;
        goto end;
    }
    // record end position of audio tokens for later memory reset
    endPos = *pCurrentPos;

    // eval generation prompt
    ret = tokenize(lccCtx->mtmdContext, generationPrompt, NULL, 0, chunks);
    if (ret != LCC_ERROR_SUCCESS) {
        goto end;
    }
    metric_start("== EVAL GENERATION PROMPT");
    r = mtmd_helper_eval_chunks(
        lccCtx->mtmdContext,
        lccCtx->ctx,
        chunks,
        *pCurrentPos,
        0,
        llama_n_batch(lccCtx->ctx),
        true,
        pCurrentPos
    );
    metric_end();
    if (r != 0) {
        ret = LCC_ERROR_FAILED_MTMD_EVAL;
        goto end;
    }

    // do prediction to get the output text
    size_t lenOutputBuf = 4096;
    size_t indexOutputBuf = 0;
    char *outputBuf = calloc(lenOutputBuf, sizeof(*outputBuf));
    if (!outputBuf) {
        // impossible
        ret = LCC_ERROR_FAILED_ALLOCATE_MEMORY;
        goto end;
    }

    for (; *pCurrentPos < (llama_pos)llama_n_ctx(lccCtx->ctx); (*pCurrentPos)++) {
        // sample last token
        const llama_token id = llama_sampler_sample(lccCtx->sampler, lccCtx->ctx, -1);
        if (id == 0) {
            // failed to sample token, end generation
            ret = LCC_ERROR_FAILED_SAMPLE;
            goto end;
        }

        llama_sampler_accept(lccCtx->sampler, id);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        int32_t lenPiece;
retry:
        lenPiece = llama_token_to_piece(vocab, id, &outputBuf[indexOutputBuf], lenOutputBuf - indexOutputBuf, 0, true);
        if (lenPiece < 0) {
            // buffer is too small to hold the piece, extend it
            lenOutputBuf *= 2;
            char *newBuf = realloc(outputBuf, lenOutputBuf);
            if (!newBuf) {
                // impossible
                ret = LCC_ERROR_FAILED_ALLOCATE_MEMORY;
                goto end;
            }
            outputBuf = newBuf;
            goto retry;
        }
        indexOutputBuf += lenPiece;

        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = *pCurrentPos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        metric_start("== DECODE TOKEN");
        r = llama_decode(lccCtx->ctx, batch);
        if (r != 0) {
            free(outputBuf);
            ret = LCC_ERROR_FAILED_LLM_DECODE;
            goto end;
        }
        metric_end();
    }

    *callerFreeOutputBuffer = outputBuf;
    // reset position for next round of ASR
    llama_memory_seq_rm(mem, 0, endPos, *pCurrentPos);
    *pCurrentPos = endPos;

end:
    llama_batch_free(batch);
    if (chunks) {
        mtmd_input_chunks_free(chunks);
    }
    return ret;
}

static LCCErrorCode qwen3ASRRtrimSymbols(
    const struct llama_vocab *vocab,
    size_t rTrimSymbols,
    llama_token *tokens,
    size_t *pNTokens
) {
    LCCErrorCode ret;
    int32_t n;
    int32_t endIndex;
    char pieceBuf[4096];

    *pNTokens -= ((*pNTokens) > rTrimSymbols) ? rTrimSymbols : (*pNTokens); // force remove at most rTrimSymbols tokens

    // strip tokens ends with symbols from the end
    for (endIndex = (*pNTokens) - 1; endIndex >= 0; endIndex--) {
        llama_token token = tokens[endIndex];
        int32_t n = llama_token_to_piece(vocab, token, pieceBuf, sizeof(pieceBuf), 0, true);
        if (n <= 0) {
            ret = LCC_ERROR_FAILED_DETOKENIZE;
            goto end;
        }

        bool isSymbol = false;
        switch (pieceBuf[n - 1]) {
            case '.':
            case ',':
            case '?':
            case '!':
            case '\r':
            case '\n':
            case '\t':
                isSymbol = true;
                continue;
        }
        if (n >= 3) {
            if (
#define checkWithSymbol(str) \
                (pieceBuf[n - 1] == str[2] && pieceBuf[n - 2] == str[1] && pieceBuf[n - 3] == str[0])
                checkWithSymbol("\u3001") || // 、 IDEOGRAPHIC COMMA
                checkWithSymbol("\u3002") || // 。 IDEOGRAPHIC FULL STOP
                checkWithSymbol("\uff01") || // ！ FULLWIDTH EXCLAMATION MARK
                checkWithSymbol("\uff0c") || // ， FULLWIDTH COMMA
                checkWithSymbol("\uff1a") || // ： FULLWIDTH COLON
                checkWithSymbol("\uff1b") || // ； FULLWIDTH SEMICOLON
#undef checkWithSymbol
                false
            ) {
                isSymbol = true;
            }
        }
        if (!isSymbol) {
            break;
        }
    }

    *pNTokens = endIndex + 1;

    ret = LCC_ERROR_SUCCESS;
end:
    return ret;
}

// this function will decode tokens like:
// <|im_start|>system\n(newPrompt)<|im_end|>\n<|im_start|>user\n<|audio_start|>
// return new position by pNewPos
// returns LCC_ERROR_SUCCESS if success, otherwise error code
LCCErrorCode LCCStreamingASRQwen3ASRSetPrompt(
    LCCContext *lccCtx,
    int32_t *pNewPos,
    const char *newPrompt,
    size_t sizeNewPrompt
) {
    LCCErrorCode ret;
    const struct llama_vocab *vocab = llama_model_get_vocab(lccCtx->model);
    int32_t nTokens;
    llama_token *tokens = NULL;
    nTokens = llama_tokenize(vocab, newPrompt, (int32_t)sizeNewPrompt, NULL, 0, false, true);
    // if (nTokens == 0) {
    //     ret = LCC_ERROR_FAILED_TOKENIZE;
    //     goto end;
    // }
    tokens = malloc((3 - nTokens + 7) * sizeof(*tokens));
    if (!tokens) {
        ret = LCC_ERROR_FAILED_ALLOCATE_MEMORY;
        goto end;
    }
    tokens[0] = 151644; // <|im_start|>
    tokens[1] = 8948;   // system
    tokens[2] = 198;    // \n
    nTokens = llama_tokenize(vocab, newPrompt, sizeNewPrompt, tokens + 3, -nTokens, false, true);
    // if (nTokens <= 0) {
    //     ret = LCC_ERROR_FAILED_TOKENIZE;
    //     goto end;
    // }
    tokens[3 + nTokens] = 198; // \n
    tokens[3 + nTokens + 1] = 151645; // <|im_end|>
    tokens[3 + nTokens + 2] = 198; // \n
    tokens[3 + nTokens + 3] = 151644; // <|im_start|>
    tokens[3 + nTokens + 4] = 872; // user
    tokens[3 + nTokens + 5] = 198; // \n
    tokens[3 + nTokens + 6] = 151669; // <|audio_start|>

    // clean memory
    llama_memory_clear(llama_get_memory(lccCtx->ctx), false);

    // eval tokens
    *pNewPos = 0;
    ret = decodeAsBatch(lccCtx, tokens, 3 + nTokens + 7, pNewPos, false);

end:
    if (tokens) {
        free(tokens);
    }
    return ret;
}

// this function will
// 1. tokenize audio chunks into embed (mtmd chunks)
// 2. eval audio chunks to update LLM memory, record the position
// 3. right trim calleeAllocateCallerFreeOutputTokens to remove tokens at end with symbols endings (。，things)
// 4. predict until <asr_text>
//   a. if generation changes before <asr_text>, regenerate all tokens ignoring calleeAllocateCallerFreeOutputTokens
//   b. if generation reaches <asr_text> and tokens matched, prefill remaining tokens in calleeAllocateCallerFreeOutputTokens
// 5. until EOG, stop generation
// 6. remove memory from position recorded in step 2 to the end
// 7. return output tokens in calleeAllocateCallerFreeOutputTokens, and set *pCurrentPos to the position after audio tokens for next round of ASR
// pass calleeAllocateCallerFreeOutputTokens as pointer to NULL and pOutputTokenCount as pointer to 0 at first call
LCCErrorCode LCCStreamingASRQwen3ASRFeedSamples(
    LCCContext *lccCtx,
    int32_t *pCurrentPos,
    const float *pcmData,
    size_t nSamples,
    llama_token **calleeAllocateCallerFreeOutputTokens,
    size_t *pOutputTokenCount,
    size_t rTrimTokens
) {
    LCCErrorCode ret = LCC_ERROR_SUCCESS;
    int32_t r;
    int32_t endPos;
    const struct llama_vocab *vocab = llama_model_get_vocab(lccCtx->model);
    mtmd_input_chunks *chunks;
    size_t nOutputSize = 0;
    llama_memory_t mem = llama_get_memory(lccCtx->ctx);
    llama_batch batch = llama_batch_init(1, 0, 1);

    chunks = mtmd_input_chunks_init();
    if (!chunks) {
        ret = LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS;
        goto end;
    }

    // do mtmd tokenization for audio tokens
    ret = tokenize(lccCtx->mtmdContext, mtmd_default_marker(), pcmData, nSamples, chunks);
    if (ret != LCC_ERROR_SUCCESS) {
        goto end;
    }

    // eval new audio chunk
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            continue;
        }

        metric_start("== EVAL NEW AUDIO CHUNK");
        r = mtmd_helper_eval_chunk_single(
            lccCtx->mtmdContext,
            lccCtx->ctx,
            chunk,
            *pCurrentPos,
            0,
            llama_n_batch(lccCtx->ctx),
            true,
            pCurrentPos
        );
        metric_end();
        if (r != 0) {
            ret = LCC_ERROR_FAILED_MTMD_EVAL;
            goto end;
        }
    }
    mtmd_input_chunks_free(chunks);
    chunks = mtmd_input_chunks_init(); // re-init for generation prompt
    if (!chunks) {
        ret = LCC_ERROR_FAILED_CREATE_MTMD_INPUT_CHUNKS;
        goto end;
    }
    // record end position of audio tokens for later memory reset
    endPos = *pCurrentPos;

    // decode generate prompts
    const llama_token generatePromptTokens[] = {
        151670, // <|audio_end|>
        151645, // <|im_end|>
        198, // \n
        151644, // <|im_start|>
        77091, // assistant
        198, // \n
        11528, // language
    };
    ret = decodeAsBatch(
        lccCtx,
        generatePromptTokens,
        sizeof(generatePromptTokens) / sizeof(*generatePromptTokens),
        pCurrentPos,
        true
    );
    if (ret != LCC_ERROR_SUCCESS) {
        goto end;
    }

    // prepare output tokens buffer
#define outputTokens (*calleeAllocateCallerFreeOutputTokens)
#define nOutputTokens (*pOutputTokenCount)
    if (outputTokens == NULL || nOutputTokens == 0) {
        // allocate output buffer
        nOutputTokens = 0;
        nOutputSize = 1024;
        outputTokens = malloc(nOutputSize * sizeof(*outputTokens));
        if (!outputTokens) {
            // impossible
            ret = LCC_ERROR_FAILED_ALLOCATE_MEMORY;
            goto end;
        }
    } else {
        // already allocated by self
        // align size
        nOutputSize = 1024;
        while (nOutputSize < nOutputTokens) {
            nOutputSize *= 2;
        }
    }

    // right trim output tokens
    ret = qwen3ASRRtrimSymbols(
        (struct llama_vocab *)vocab,
        rTrimTokens,
        outputTokens,
        &nOutputTokens
    );
    if (ret != LCC_ERROR_SUCCESS) {
        goto end;
    }

    // do prediction to get the output tokens
    size_t outputTokenIndex = 0;
    bool newGenerate = false;
    for (; *pCurrentPos < (llama_pos)llama_n_ctx(lccCtx->ctx); (*pCurrentPos)++) {
        // sample last token
        const llama_token id = llama_sampler_sample(lccCtx->sampler, lccCtx->ctx, -1);
        if (id == 0) {
            // failed to sample token, end generation
            ret = LCC_ERROR_FAILED_SAMPLE;
            goto end;
        }
        llama_sampler_accept(lccCtx->sampler, id);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        if (!newGenerate) {
            // at 'language'
            if (outputTokenIndex >= nOutputTokens || id != outputTokens[outputTokenIndex]) {
                // language changed, no more prefill
                newGenerate = true;
            } else if (id == 151704) {
                // at '<asr_text>', prefill remaining tokens
                if (outputTokenIndex < nOutputTokens) {
                    // prefill remaining tokens in output buffer
                    ret = decodeAsBatch(
                        lccCtx,
                        outputTokens + outputTokenIndex,
                        nOutputTokens - outputTokenIndex,
                        pCurrentPos,
                        true
                    );
                    if (ret != LCC_ERROR_SUCCESS) {
                        goto end;
                    }
                    outputTokenIndex = nOutputTokens; // all tokens are prefilled
                }
                newGenerate = true;
                (*pCurrentPos)--;
                continue; // continue to sample next tokens
            }
            // otherwise, continue to generate
        }

        // record output token
        if (outputTokenIndex >= nOutputSize) {
            // need more space to hold output tokens, extend it
            nOutputSize *= 2;
            llama_token *newBuf = realloc(outputTokens, nOutputSize * sizeof(*newBuf));
            if (!newBuf) {
                // impossible
                ret = LCC_ERROR_FAILED_ALLOCATE_MEMORY;
                goto end;
            }
            outputTokens = newBuf;
        }
        outputTokens[outputTokenIndex++] = id;

        // decode token
        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = *pCurrentPos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        metric_start("== DECODE TOKEN");
        r = llama_decode(lccCtx->ctx, batch);
        if (r != 0) {
            ret = LCC_ERROR_FAILED_LLM_DECODE;
            goto end;
        }
        metric_end();
    }
#undef outputTokens
#undef nOutputTokens

    // reset position for next round of ASR
    llama_memory_seq_rm(mem, 0, endPos, *pCurrentPos);
    *pCurrentPos = endPos;
    *pOutputTokenCount = outputTokenIndex;

end:
    llama_batch_free(batch);
    if (chunks) {
        mtmd_input_chunks_free(chunks);
    }
    return ret;
}
