
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>

#include "wrapper.h"

void LCCDefaultContextInitParams(LCCContextInitParams *initParams)
{
    memset(initParams, 0, sizeof(LCCContextInitParams));

    initParams->modelParams = llama_model_default_params();
    initParams->ctxParams = llama_context_default_params();
}

LCCErrorCode LCCInitContext(LCCContext *lccCtx, const LCCContextInitParams *initParams)
{
    LCCErrorCode ret = LCC_ERROR_SUCCESS;
    struct llama_model_params modelParams;
    struct llama_context_params ctxParams;

    memset(lccCtx, 0, sizeof(LCCContext));

    // open model
    if (!initParams->modelPath) {
        ret = LCC_ERROR_INVALID_ARGUMENT;
        goto end;
    }

    lccCtx->model = llama_model_load_from_file(initParams->modelPath, initParams->modelParams);
    if (!lccCtx->model) {
        ret = LCC_ERROR_FAILED_LOAD_MODEL;
        goto end;
    }

    // create llamacpp context
    lccCtx->ctx = llama_init_from_model(lccCtx->model, initParams->ctxParams);
    if (!lccCtx->ctx) {
        ret = LCC_ERROR_FAILED_CREATE_CONTEXT;
        goto end;
    }

    // create sampler
    // nothing to set in llama_sampler_chain_params
    struct llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    lccCtx->sampler = llama_sampler_chain_init(samplerParams);
    if (!lccCtx->sampler) {
        fprintf(stderr, "Failed to create sampler\n");
        ret = LCC_ERROR_FAILED_CREATE_SAMPLER;
        goto end;
    }
    switch (initParams->samplerConfig.kind) {
        case LCC_SAMPLER_GREEDY:
            llama_sampler_chain_add(lccCtx->sampler, llama_sampler_init_greedy());
            break;
        case LCC_SAMPLER_DIST:
            llama_sampler_chain_add(lccCtx->sampler, llama_sampler_init_top_k(initParams->samplerConfig.config.dist.top_k));
            llama_sampler_chain_add(lccCtx->sampler, llama_sampler_init_top_p(initParams->samplerConfig.config.dist.top_p, 1));
            llama_sampler_chain_add(lccCtx->sampler, llama_sampler_init_temp(initParams->samplerConfig.config.dist.temperature));
            uint32_t seed = initParams->samplerConfig.config.dist.seed;
            if (initParams->samplerConfig.config.dist.seed == 0) {
                // seed it with random
                // TODO: use /dev/random
                getrandom(&seed, sizeof(seed), 0);
            }
            llama_sampler_chain_add(lccCtx->sampler, llama_sampler_init_dist(seed));
            break;
        default:
            ret = LCC_ERROR_NOT_IMPLEMENTED;
            goto end;
    }

    if (initParams->mmprojPath) {
        // initialize mtmd context
        struct mtmd_context_params mtmdCtxParams = mtmd_context_params_default();
        lccCtx->mtmdContext = mtmd_init_from_file(initParams->mmprojPath, lccCtx->model, mtmdCtxParams);
        if (!lccCtx->mtmdContext) {
            ret = LCC_ERROR_FAILED_CREATE_MTMD_CONTEXT;
            goto end;
        }
    }

end:
    if (ret != LCC_ERROR_SUCCESS) {
        LCCFreeContext(lccCtx);
    }

    return ret;
}

void LCCFreeContext(LCCContext *lccCtx)
{
    if (lccCtx->mtmdContext) {
        mtmd_free(lccCtx->mtmdContext);
        lccCtx->mtmdContext = NULL;
    }
    if (lccCtx->sampler) {
        llama_sampler_free(lccCtx->sampler);
        lccCtx->sampler = NULL;
    }
    if (lccCtx->ctx) {
        llama_free(lccCtx->ctx);
        lccCtx->ctx = NULL;
    }
    if (lccCtx->model) {
        llama_model_free(lccCtx->model);
        lccCtx->model = NULL;
    }
}
