import struct
from pathlib import Path

import numpy
import typer


def _ensure_c4_dataset(data_path: str, seq_length: int = 8192) -> None:
    prefix = Path(data_path)
    bin_path = prefix.with_suffix(".bin")
    idx_path = prefix.with_suffix(".idx")
    if bin_path.exists() and idx_path.exists():
        with open(idx_path, "rb") as f:
            header = f.read(9)
            if header == b"MMIDIDX\x00\x00":
                version = struct.unpack("<Q", f.read(8))[0]
                if version == 1:
                    f.read(1)
                    seq_count = struct.unpack("<Q", f.read(8))[0]
                    if seq_count >= 400:
                        return
        typer.echo("Existing C4 dataset is outdated, regenerating...")
        bin_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise typer.BadParameter(
            "C4 dataset requires 'transformers' and 'datasets' libraries. "
            "Install them with: uv pip install transformers datasets"
        ) from exc

    typer.echo("Downloading C4/en from HuggingFace and tokenizing with Llama-3-8B tokenizer ...")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
    vocab_size = tokenizer.vocab_size

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    num_docs = 1000
    token_ids: list[numpy.ndarray] = []
    lengths: list[int] = []
    for i, example in enumerate(ds):
        if i >= num_docs:
            break
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        if len(tokens) < 10:
            continue
        arr = numpy.array(tokens, dtype=numpy.int32)
        token_ids.append(arr)
        lengths.append(len(arr))

    if len(lengths) < 400:
        raise typer.BadParameter(
            f"Only got {len(lengths)} valid C4 documents; "
            "the dataset stream may have been empty or truncated."
        )

    with open(bin_path, "wb") as f:
        for arr in token_ids:
            f.write(arr.tobytes(order="C"))

    pointers: list[int] = []
    curr = numpy.int64(0)
    itemsize = numpy.dtype(numpy.int32).itemsize
    for length in lengths:
        pointers.append(curr.item())
        curr += length * itemsize

    document_indices = list(range(len(lengths) + 1))

    with open(idx_path, "wb") as f:
        f.write(b"MMIDIDX\x00\x00")
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))
        f.write(struct.pack("<Q", len(lengths)))
        f.write(struct.pack("<Q", len(document_indices)))
        f.write(numpy.array(lengths, dtype=numpy.int32).tobytes(order="C"))
        f.write(numpy.array(pointers, dtype=numpy.int64).tobytes(order="C"))
        f.write(numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C"))

    typer.echo(f"Dataset ready: {bin_path}, {idx_path} ({len(lengths)} docs, vocab={vocab_size})")
