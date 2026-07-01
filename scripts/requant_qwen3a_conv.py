#!/usr/bin/env python3
# Experimental: requantize Qwen3-ASR mmproj conv2d.2/3 weights to Q8_0, stored
# pre-reshaped to 2D [IC*KH*KW, OC] so the Vulkan integer-dot matmul path can be
# used (needs ne0 % 32 == 0). conv2d.1 (k=9) stays untouched.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType  # noqa: E402

REQUANT = {"a.conv2d.2.weight", "a.conv2d.3.weight"}


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: requant_qwen3a_conv.py <in.gguf> <out.gguf>")
        sys.exit(1)

    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = GGUFReader(src, "r")

    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch.contents() if arch else "clip"
    writer = GGUFWriter(dst, arch)

    # copy metadata
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # copy / requantize tensors
    for t in reader.tensors:
        if t.name in REQUANT:
            arr = np.asarray(t.data).astype(np.float32)   # numpy shape (OC, IC, KH, KW)
            oc = arr.shape[0]
            arr = np.ascontiguousarray(arr.reshape(oc, -1))  # (OC, IC*KH*KW), KW fastest
            q = gguf.quants.quantize(arr, GGMLQuantizationType.Q8_0)
            writer.add_tensor(t.name, q, raw_dtype=GGMLQuantizationType.Q8_0)
            print(f"requant {t.name}: {list(t.shape)} F16 -> [{arr.shape[1]}, {oc}] Q8_0")
        elif t.data.dtype == np.uint8:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        else:
            writer.add_tensor(t.name, t.data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
