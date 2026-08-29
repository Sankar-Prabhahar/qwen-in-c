#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "sampler.h"
#include "softmax.h"

static float random_f32(uint64_t *state)
{
    /* xorshift64star */
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    uint64_t val = *state * 0x2545F4914F6CDD1DULL;
    return (float)(val >> 11) * (1.0f / 9007199254740992.0f);
}

typedef struct {
    int id;
    float prob;
} ProbIndex;

static int compare_probs(const void *a, const void *b)
{
    float pa = ((const ProbIndex *)a)->prob;
    float pb = ((const ProbIndex *)b)->prob;
    if (pb > pa) return 1;
    if (pb < pa) return -1;
    return 0;
}

int sample_token(float *logits, int vocab_size, SamplerConfig *config)
{
    if (!logits || vocab_size <= 0) return 0;

    /* 1. Greedy Argmax */
    if (!config || config->temperature <= 0.0f || config->top_k == 1) {
        int best_id = 0;
        float best_logit = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    /* 2. Temperature scaling */
    float inv_temp = 1.0f / config->temperature;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }

    /* 3. Softmax to get probabilities */
    softmax(logits, vocab_size);

    /* 4. Top-K filtering */
    int k = (config->top_k > 0 && config->top_k < vocab_size) ? config->top_k : vocab_size;
    ProbIndex *probs = (ProbIndex *)malloc(vocab_size * sizeof(ProbIndex));
    if (!probs) return 0;

    for (int i = 0; i < vocab_size; i++) {
        probs[i].id = i;
        probs[i].prob = logits[i];
    }

    qsort(probs, vocab_size, sizeof(ProbIndex), compare_probs);

    /* 5. Top-P (nucleus) truncation */
    float cumulative = 0.0f;
    int cutoff = k;
    float top_p = (config->top_p > 0.0f && config->top_p <= 1.0f) ? config->top_p : 1.0f;

    for (int i = 0; i < k; i++) {
        cumulative += probs[i].prob;
        if (cumulative >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* 6. Renormalize & Sample */
    float r = random_f32(&config->rng_seed) * cumulative;
    float running = 0.0f;
    int selected = probs[0].id;

    for (int i = 0; i < cutoff; i++) {
        running += probs[i].prob;
        if (r <= running) {
            selected = probs[i].id;
            break;
        }
    }

    free(probs);
    return selected;
}
