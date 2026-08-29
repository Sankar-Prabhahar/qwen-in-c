#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

#include "model_engine.h"
#include "sampler.h"

static void run_deterministic_test(Engine *engine,
                                  const char *title,
                                  const int *prompt_tokens,
                                  int prompt_len,
                                  int max_new_tokens)
{
    printf("========================================================\n");
    printf(" TEST: \"%s\"\n", title);
    printf("========================================================\n");

    ModelKVCache cache;
    int max_seq = prompt_len + max_new_tokens + 64;
    kv_cache_init(&cache, engine->n_layers, max_seq, engine->n_kv_heads, engine->head_dim);

    float *logits = (float *)malloc(engine->vocab_size * sizeof(float));
    if (!logits) {
        kv_cache_free(&cache);
        return;
    }

    printf("Prompt tokens (%d): [", prompt_len);
    for (int i = 0; i < prompt_len; i++) {
        printf("%d%s", prompt_tokens[i], (i < prompt_len - 1) ? ", " : "]\n");
    }

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    /* 1. Ingest Prompt Tokens */
    QueryPerformanceCounter(&start);
    for (int i = 0; i < prompt_len; i++) {
        engine_forward(engine, &cache, prompt_tokens[i], i, logits);
    }
    QueryPerformanceCounter(&end);
    double prompt_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    printf("Prompt ingestion completed in %.2f ms (%.2f ms/token)\n\n",
           prompt_ms, prompt_ms / (double)prompt_len);

    /* 2. Top-10 Logits for the Next Token */
    engine_print_top_logits(engine, logits, 10);

    /* 3. Greedy Autoregressive Generation (temp=0) */
    SamplerConfig greedy_cfg = {
        .temperature = 0.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .rng_seed = 0
    };

    printf("\n  Deterministic Generation (temp=0):\n  Output: \"");

    /* Print reconstructed prompt string */
    for (int i = 0; i < prompt_len; i++) {
        const char *s = tokenizer_decode(&engine->tokenizer, prompt_tokens[i]);
        /* Format token string inline */
        if (strcmp(s, "<s>") != 0) {
            const unsigned char *u = (const unsigned char *)s;
            while (*u) {
                if (u[0] == 0xE2 && u[1] == 0x96 && u[2] == 0x81) {
                    putchar(' '); u += 3;
                } else {
                    putchar(*u++);
                }
            }
        }
    }

    int next_token = sample_token(logits, engine->vocab_size, &greedy_cfg);

    QueryPerformanceCounter(&start);
    for (int step = 0; step < max_new_tokens; step++) {
        int pos = prompt_len + step;
        if (next_token == engine->tokenizer.eos_id) break;

        const char *s = tokenizer_decode(&engine->tokenizer, next_token);
        const unsigned char *u = (const unsigned char *)s;
        while (*u) {
            if (u[0] == 0xE2 && u[1] == 0x96 && u[2] == 0x81) {
                putchar(' '); u += 3;
            } else if (u[0] == '<' && u[1] == '0' && u[2] == 'x' && u[5] == '>') {
                unsigned int b = 0;
                if (sscanf((const char *)u, "<0x%02X>", &b) == 1) {
                    putchar((char)b); u += 6;
                } else {
                    putchar(*u++);
                }
            } else {
                putchar(*u++);
            }
        }
        fflush(stdout);

        if (!engine_forward(engine, &cache, next_token, pos, logits)) break;
        next_token = sample_token(logits, engine->vocab_size, &greedy_cfg);
    }
    QueryPerformanceCounter(&end);
    double gen_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    printf("\"\n  (Generation: %d tokens in %.2f ms, ~%.2f ms/token)\n\n",
           max_new_tokens, gen_ms, gen_ms / (double)max_new_tokens);

    free(logits);
    kv_cache_free(&cache);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("========================================\n");
    printf("   Qwen-in-C Deterministic Test Suite   \n");
    printf("========================================\n\n");

    const char *model_path = (argc >= 2) ? argv[1] : "tinyllama.gguf";

    Engine *engine = (Engine *)calloc(1, sizeof(Engine));
    if (!engine) return 1;

    if (!engine_init(engine, model_path)) {
        printf("Failed to init engine\n");
        free(engine);
        return 1;
    }

    printf("Engine initialized: %d layers, %d hidden_dim, %d heads, %d kv_heads, vocab_size: %d (OpenMP Threads: %d)\n\n",
           engine->n_layers, engine->hidden_dim, engine->n_heads, engine->n_kv_heads, engine->vocab_size, omp_get_max_threads());

    /* Test 1: "Hello" */
    int prompt_hello[] = {1, 15043};
    run_deterministic_test(engine, "Hello", prompt_hello, sizeof(prompt_hello)/sizeof(int), 15);

    /* Test 2: "The capital of France is" */
    int prompt_france[] = {1, 450, 7483, 310, 3444, 338};
    run_deterministic_test(engine, "The capital of France is", prompt_france, sizeof(prompt_france)/sizeof(int), 15);

    /* Test 3: "Once upon a time" */
    int prompt_story[] = {1, 9038, 2501, 263, 931};
    run_deterministic_test(engine, "Once upon a time", prompt_story, sizeof(prompt_story)/sizeof(int), 15);

    engine_free(engine);
    free(engine);
    printf("All deterministic tests finished successfully.\n");
    return 0;
}