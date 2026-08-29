#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <omp.h>

#include "model_engine.h"
#include "sampler.h"

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("========================================\n");
    printf("   Qwen-in-C Pure C Inference Engine   \n");
    printf("========================================\n\n");

    const char *model_path = (argc >= 2) ? argv[1] : "tinyllama.gguf";
    printf("Loading model: %s ...\n", model_path);

    Engine *engine = (Engine *)calloc(1, sizeof(Engine));
    if (!engine) {
        printf("Failed to allocate Engine memory\n");
        return 1;
    }

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    if (!engine_init(engine, model_path)) {
        printf("Failed to initialize engine with model %s\n", model_path);
        free(engine);
        return 1;
    }
    QueryPerformanceCounter(&end);

    double load_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    printf("Model loaded & indexed in %.2f ms (Threads: %d)\n", load_ms, omp_get_max_threads());
    printf("Architecture: %d layers, %d hidden_dim, %d heads, %d kv_heads, vocab_size: %d\n\n",
           engine->n_layers, engine->hidden_dim, engine->n_heads, engine->n_kv_heads, engine->vocab_size);

    /* Test 1: "The capital of France is" */
    int prompt_france[] = {1, 450, 7483, 310, 3444, 338};
    int prompt_len = sizeof(prompt_france) / sizeof(prompt_france[0]);

    SamplerConfig greedy_config = {
        .temperature = 0.0f,
        .top_k = 1,
        .top_p = 1.0f,
        .rng_seed = 42
    };

    printf("--- [Generation Test: Greedy Decoding (10 new tokens)] ---\n");
    printf("Prompt: \"The capital of France is\"\nOutput: ");
    fflush(stdout);

    QueryPerformanceCounter(&start);
    engine_generate(engine, prompt_france, prompt_len, 10, &greedy_config);
    QueryPerformanceCounter(&end);

    double gen_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    printf("(Total time: %.2f s, ~%.2f ms/token)\n\n", gen_ms / 1000.0, gen_ms / 10.0);

    engine_free(engine);
    free(engine);
    printf("Engine shut down cleanly.\n");
    return 0;
}