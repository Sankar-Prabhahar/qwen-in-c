#ifndef FFN_H
#define FFN_H

#include "model.h"
#include "tensor.h"

/*
 * ffn_swiglu:
 * Compute SwiGLU Feed-Forward Network:
 *   gate = W_gate * x
 *   up   = W_up * x
 *   act  = silu(gate) * up
 *   out  = W_down * act
 *
 * Parameters:
 *   model:              Mapped GGUF model
 *   gate_tensor:        W_gate tensor
 *   up_tensor:          W_up tensor
 *   down_tensor:        W_down tensor
 *   x:                  Input vector [hidden_dim]
 *   hidden_dim:         Hidden dimension (e.g. 2048)
 *   intermediate_dim:   Intermediate FFN dimension (e.g. 5632)
 *   out:                Output vector [hidden_dim]
 */
int ffn_swiglu(const Model *model,
               const Tensor *gate_tensor,
               const Tensor *up_tensor,
               const Tensor *down_tensor,
               const float *x,
               int hidden_dim,
               int intermediate_dim,
               float *out);

#endif
