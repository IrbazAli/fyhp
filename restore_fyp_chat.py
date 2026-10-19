#!/usr/bin/env python3
import sqlite3
import base64
import io
import os
import sys

def encode_varint(val):
    res = bytearray()
    while True:
        b = val & 0x7f
        val >>= 7
        if val:
            res.append(b | 0x80)
        else:
            res.append(b)
            break
    return bytes(res)

def encode_field(field_num, wire_type, data):
    tag = (field_num << 3) | wire_type
    if wire_type == 0:
        return encode_varint(tag) + encode_varint(data)
    elif wire_type == 2:
        return encode_varint(tag) + encode_varint(len(data)) + data
    else:
        raise ValueError(f'unsupported wire {wire_type}')

def read_varint(stream):
    res = 0
    shift = 0
    while True:
        b = stream.read(1)
        if not b:
            return None
        byte = b[0]
        res |= (byte & 0x7f) << shift
        if not (byte & 0x80):
            return res
        shift += 7

state_db_path = os.path.expanduser('~/.config/Antigravity IDE/User/globalStorage/state.vscdb')

if not os.path.exists(state_db_path):
    print(f"Error: {state_db_path} does not exist.")
    sys.exit(1)

conn = sqlite3.connect(state_db_path)
c = conn.cursor()
c.execute("SELECT value FROM ItemTable WHERE key = 'antigravityUnifiedStateSync.trajectorySummaries'")
row = c.fetchone()
if not row:
    print("Error: trajectorySummaries key not found in database.")
    sys.exit(1)

raw = base64.b64decode(row[0])

entries = {}
s = io.BytesIO(raw)
while True:
    k = read_varint(s)
    if k is None:
        break
    w = k & 7
    f = k >> 3
    if w == 2:
        l = read_varint(s)
        d = s.read(l)
        sub = io.BytesIO(d)
        cid = None
        val = None
        while True:
            k2 = read_varint(sub)
            if k2 is None:
                break
            w2 = k2 & 7
            f2 = k2 >> 3
            if w2 == 2:
                l2 = read_varint(sub)
                d2 = sub.read(l2)
                if f2 == 1:
                    cid = d2.decode('utf-8', errors='ignore')
                elif f2 == 2:
                    val = d2
        if cid and val:
            entries[cid] = val

# Build FYP summary
traj_meta = (
    encode_field(1, 2, b'\n(file:///home/irbazali/Desktop/akashatech\x1a\x00') +
    encode_field(1, 2, b'\n!file:///home/irbazali/Desktop/FYP\x1a\x00') +
    encode_field(2, 2, b'\x08\xb7\xdf\xa2\xd6\x06\x10\xd4\x95\xd8\xe2\x01') +
    encode_field(3, 2, b'3dbdbca5-4241-4087-8ef5-3e9bb429960f') +
    encode_field(6, 2, b'74342d71-8fa5-4dac-9181-b9d28c1bf924') +
    encode_field(7, 2, b'file:///home/irbazali/Desktop/akashatech') +
    encode_field(7, 2, b'file:///home/irbazali/Desktop/FYP')
)

summary_bytes = (
    encode_field(1, 2, 'Aasaan Khata (FYP)'.encode('utf-8')) +
    encode_field(2, 0, 993) +
    encode_field(3, 2, b'\x08\xef\xb7\xa3\xd6\x06\x10\x87\xfb\xe4F') +
    encode_field(4, 2, b'c7a164e4-0cc1-4c81-ae7b-bd54fa0c2b0d') +
    encode_field(5, 0, 1) +
    encode_field(7, 2, b'\x08\xb7\xdf\xa2\xd6\x06\x10\x95\xc6\xd4\xfb\x01') +
    encode_field(9, 2, b'\n(file:///home/irbazali/Desktop/akashatech\x1a\x00') +
    encode_field(9, 2, b'\n!file:///home/irbazali/Desktop/FYP\x1a\x00') +
    encode_field(10, 2, b'\x08\xdf\xb7\xa3\xd6\x06\x10\x9d\x99\xd98') +
    encode_field(15, 2, b'') +
    encode_field(16, 0, 991) +
    encode_field(17, 2, traj_meta) +
    encode_field(22, 0, 4)
)

b64_summary = base64.b64encode(summary_bytes).decode('ascii')
val_msg = encode_field(1, 2, b64_summary.encode('ascii'))

entries['74342d71-8fa5-4dac-9181-b9d28c1bf924'] = val_msg

outer_bytes = bytearray()
for cid, val in entries.items():
    entry_pair = encode_field(1, 2, cid.encode('utf-8')) + encode_field(2, 2, val)
    outer_bytes.extend(encode_field(1, 2, entry_pair))

new_b64_val = base64.b64encode(outer_bytes).decode('ascii')

c.execute("UPDATE ItemTable SET value = ? WHERE key = 'antigravityUnifiedStateSync.trajectorySummaries'", (new_b64_val,))
conn.commit()
conn.close()

print("Successfully injected 'Aasaan Khata (FYP)' into conversation index!")
print("Registered conversations in index:")
for cid in entries.keys():
    print(f" - {cid}")
