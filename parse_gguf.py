"""
Parse GGUF metadata with correct type enum to find tensor data_start.
GGUF v2 types: 0=uint8,1=int8,2=uint16,3=int16,4=uint32,5=int32,
               6=float32,7=bool,8=string,9=array,10=uint64,11=int64,12=float64
"""
import struct
import sys

def parse_gguf(path):
    with open(path, 'rb') as f:
        data = f.read()

    magic = data[:4]
    pos = 4
    version = struct.unpack_from('<I', data, pos)[0]; pos += 4
    tensor_count = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    metadata_count = struct.unpack_from('<Q', data, pos)[0]; pos += 8

    print(f"Magic: {magic}, v{version}, tensors={tensor_count}, meta={metadata_count}")

    def skip_value(pos, vtype):
        if vtype in (0, 1, 7):
            return pos + 1
        elif vtype in (2, 3):
            return pos + 2
        elif vtype in (4, 5, 6):
            return pos + 4
        elif vtype in (10, 11, 12):
            return pos + 8
        elif vtype == 8:  # string
            slen = struct.unpack_from('<Q', data, pos)[0]
            return pos + 8 + slen
        elif vtype == 9:  # array
            elem_type = struct.unpack_from('<I', data, pos)[0]
            count = struct.unpack_from('<Q', data, pos + 4)[0]
            pos2 = pos + 12
            if elem_type == 8:
                for _ in range(count):
                    slen = struct.unpack_from('<Q', data, pos2)[0]
                    pos2 += 8 + slen
                return pos2
            elif elem_type in (0, 1, 7):
                return pos2 + count
            elif elem_type in (2, 3):
                return pos2 + count * 2
            elif elem_type in (4, 5, 6):
                return pos2 + count * 4
            elif elem_type in (10, 11, 12):
                return pos2 + count * 8
            else:
                raise ValueError(f"Unknown array elem_type {elem_type} at {pos}")
        else:
            raise ValueError(f"Unknown vtype {vtype} at {pos}")

    for i in range(metadata_count):
        klen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        key = data[pos:pos+klen].decode('utf-8', 'replace'); pos += klen
        vtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        prev = pos
        pos = skip_value(pos, vtype)
        print(f"  [{i+1}] {key!r} vtype={vtype} size={pos-prev}")

    # Read tensor directory
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
    print(f"\nTensor directory ends at: {pos}, data_start (aligned): {data_start}")
    return data, data_start, tensors

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'tinyllama.gguf'
    data, data_start, tensors = parse_gguf(path)
    print(f"\nParsed {len(tensors)} tensors OK")

    t = tensors.get('token_embd.weight')
    if t:
        print(f"token_embd.weight: dims={t['dims']} type={t['type']} offset={t['offset']}")
