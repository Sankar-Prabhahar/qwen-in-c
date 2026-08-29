import struct

with open('tinyllama.gguf', 'rb') as f:
    data = f.read()

pos = 24
meta_count = struct.unpack_from('<Q', data, 16)[0]
tokens = []

for _ in range(meta_count):
    klen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    key = data[pos:pos+klen].decode('utf-8', 'replace'); pos += klen
    vtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
    if key == 'tokenizer.ggml.tokens':
        elem_type = struct.unpack_from('<I', data, pos)[0]; pos += 4
        count = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        for i in range(count):
            slen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
            s = data[pos:pos+slen].decode('utf-8', 'replace'); pos += slen
            tokens.append(s)
        break
    else:
        from verify_q6k import skip_value
        pos = skip_value(data, pos, vtype)

def encode_simple(text, add_bos=True):
    res = [1] if add_bos else []
    sp_text = '\u2581' + text.replace(' ', '\u2581')
    i = 0
    while i < len(sp_text):
        best_match = None
        best_len = 0
        for l in range(len(sp_text) - i, 0, -1):
            sub = sp_text[i:i+l]
            if sub in tokens:
                best_match = tokens.index(sub)
                best_len = l
                break
        if best_match is not None:
            res.append(best_match)
            i += best_len
        else:
            res.append(tokens.index(sp_text[i]))
            i += 1
    return res

prompts = [
    "Hello",
    "The capital of France is",
    "Once upon a time"
]

for p in prompts:
    ids = encode_simple(p)
    print(f"\nPrompt: {p!r}")
    print(f"Token IDs: {ids}")
    dec = [tokens[i].encode('ascii', 'backslashreplace').decode() for i in ids]
    print(f"Decoded: {dec}")
