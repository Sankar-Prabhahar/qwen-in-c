"""
Full Q6_K verification using correct GGUF parse.
Verifies our C decoder values against the Python reference.
"""
import struct
import numpy as np
import sys

QK_K = 256
BLOCK_Q6_K_SIZE = 210

def decode_q6k_row(data, offset, n_blocks):
    out = np.zeros(n_blocks * QK_K, dtype=np.float32)
    for i in range(n_blocks):
        bo = offset + i * BLOCK_Q6_K_SIZE
        ql = data[bo:bo+128]
        qh = data[bo+128:bo+192]
        sc = struct.unpack_from('<16b', data, bo+192)
        d_fp16 = struct.unpack_from('<H', data, bo+208)[0]
        d = struct.unpack('<e', struct.pack('<H', d_fp16))[0]
        base = i * QK_K

        for n_half in range(2):
            ql_off = n_half * 64
            qh_off = n_half * 32
            sc_off = n_half * 8
            out_off = base + n_half * 128

            for l in range(32):
                is_ = l // 16
                ql_l  = ql[ql_off + l]
                ql_l2 = ql[ql_off + l + 32]
                qh_l  = qh[qh_off + l]

                q1 = ((ql_l  & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32
                q2 = ((ql_l2 & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32
                q3 = ((ql_l  >>  4 ) | (((qh_l >> 4) & 3) << 4)) - 32
                q4 = ((ql_l2 >>  4 ) | (((qh_l >> 6) & 3) << 4)) - 32

                out[out_off + l]      = d * sc[sc_off + is_ + 0] * q1
                out[out_off + l + 32] = d * sc[sc_off + is_ + 2] * q2
                out[out_off + l + 64] = d * sc[sc_off + is_ + 4] * q3
                out[out_off + l + 96] = d * sc[sc_off + is_ + 6] * q4

    return out

def skip_value(data, pos, vtype):
    if vtype in (0, 1, 7): return pos + 1
    elif vtype in (2, 3): return pos + 2
    elif vtype in (4, 5, 6): return pos + 4
    elif vtype in (10, 11, 12): return pos + 8
    elif vtype == 8:
        slen = struct.unpack_from('<Q', data, pos)[0]; return pos + 8 + slen
    elif vtype == 9:
        elem_type = struct.unpack_from('<I', data, pos)[0]
        count = struct.unpack_from('<Q', data, pos + 4)[0]; pos2 = pos + 12
        if elem_type == 8:
            for _ in range(count):
                slen = struct.unpack_from('<Q', data, pos2)[0]; pos2 += 8 + slen
            return pos2
        sizes = {0:1,1:1,7:1,2:2,3:2,4:4,5:4,6:4,10:8,11:8,12:8}
        return pos2 + count * sizes.get(elem_type, 4)
    raise ValueError(f"Unknown vtype {vtype} at {pos}")

def parse_gguf(path):
    with open(path, 'rb') as f:
        data = f.read()
    pos = 4
    _ = struct.unpack_from('<I', data, pos)[0]; pos += 4
    tensor_count = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    metadata_count = struct.unpack_from('<Q', data, pos)[0]; pos += 8

    for _ in range(metadata_count):
        klen = struct.unpack_from('<Q', data, pos)[0]; pos += 8 + klen
        vtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        pos = skip_value(data, pos, vtype)

    tensors = {}
    for _ in range(tensor_count):
        nlen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        name = data[pos:pos+nlen].decode('utf-8'); pos += nlen
        ndim = struct.unpack_from('<I', data, pos)[0]; pos += 4
        dims = [struct.unpack_from('<Q', data, pos + i*8)[0] for i in range(ndim)]
        pos += ndim * 8
        ttype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        offset = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

    data_start = (pos + 31) & ~31
    return data, data_start, tensors

def main(path):
    data, data_start, tensors = parse_gguf(path)
    print(f"data_start = {data_start}")

    # --- Q decoder verification: blk.0.attn_q.weight ---
    t = tensors['blk.0.attn_q.weight']
    abs_off = data_start + t['offset']
    # Print block header so we can compare with C output
    ql = data[abs_off:abs_off+8]
    qh = data[abs_off+128:abs_off+128+8]
    sc = struct.unpack_from('<8b', data, abs_off+192)
    d_fp16 = struct.unpack_from('<H', data, abs_off+208)[0]
    d = struct.unpack('<e', struct.pack('<H', d_fp16))[0]
    print(f"\nblk.0.attn_q.weight block0:")
    print(f"  d (fp16={hex(d_fp16)}) = {d:.6f}")
    print(f"  ql[:8] = {[hex(x) for x in ql]}")
    print(f"  qh[:8] = {[hex(x) for x in qh]}")
    print(f"  sc[:8] = {list(sc)}")
    vals = decode_q6k_row(data, abs_off, 1)
    print(f"  Decoded[:8] = {vals[:8]}")

    # --- Token embedding ---
    t = tensors['token_embd.weight']
    hidden_dim = t['dims'][0]
    blocks_per_row = hidden_dim // QK_K
    row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE
    print(f"\ntoken_embd.weight: hidden={hidden_dim}, vocab={t['dims'][1]}, row_bytes={row_bytes}")

    for token_id in [0, 1, 450]:
        row_off = data_start + t['offset'] + token_id * row_bytes
        vec = decode_q6k_row(data, row_off, blocks_per_row)
        norm = float(np.linalg.norm(vec))
        print(f"  Token {token_id:4d}: norm={norm:.6f}  first6={vec[:6]}")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'tinyllama.gguf')
