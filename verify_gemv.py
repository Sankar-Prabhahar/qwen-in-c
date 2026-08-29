"""
Verify GEMV Q6_K against Python reference.
Computes W * x for blk.0.attn_q.weight with a known input vector x.
"""
import struct
import numpy as np
from verify_q6k import parse_gguf, decode_q6k_row, QK_K, BLOCK_Q6_K_SIZE

def main():
    data, data_start, tensors = parse_gguf('tinyllama.gguf')
    
    # Get token 450 embedding as our test input vector x
    t_embd = tensors['token_embd.weight']
    hidden_dim = t_embd['dims'][0]
    blocks_per_row = hidden_dim // QK_K
    row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE
    row_off = data_start + t_embd['offset'] + 450 * row_bytes
    x = decode_q6k_row(data, row_off, blocks_per_row)
    print(f"Input vector x (Token 450): norm = {np.linalg.norm(x):.4f}, first 4 = {x[:4]}")
    
    # Q projection
    t_q = tensors['blk.0.attn_q.weight']
    in_dim = t_q['dims'][0]   # 2048
    out_dim = t_q['dims'][1]  # 2048
    print(f"blk.0.attn_q.weight: in_dim={in_dim}, out_dim={out_dim}")
    
    # Compute W * x
    q_out = np.zeros(out_dim, dtype=np.float32)
    q_row_bytes = (in_dim // QK_K) * BLOCK_Q6_K_SIZE
    q_base = data_start + t_q['offset']
    
    for r in range(out_dim):
        w_row = decode_q6k_row(data, q_base + r * q_row_bytes, in_dim // QK_K)
        q_out[r] = np.dot(w_row, x)
        
    print(f"Q projection output: norm = {np.linalg.norm(q_out):.4f}")
    print(f"First 8 values of Q projection: {q_out[:8]}")

if __name__ == '__main__':
    main()
