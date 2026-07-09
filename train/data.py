"""Streaming and synthetic data iterators for Stage 0 language-model runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch

FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_NAME = "sample-10BT"
FINEWEB_SPLIT = "train"
GPT2_TOKENIZER = "gpt2"


@dataclass(frozen=True)
class FineWebDataConfig:
    seq_len: int = 1024
    dataset_path: str = FINEWEB_DATASET
    dataset_name: str = FINEWEB_NAME
    split: str = FINEWEB_SPLIT
    tokenizer_name: str = GPT2_TOKENIZER
    text_column: str = "text"
    eval_docs: int = 2_000
    add_eos: bool = True

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.eval_docs < 0:
            raise ValueError("eval_docs must be non-negative")


def pack_tokens(token_stream: Iterable[int], seq_len: int) -> Iterator[torch.Tensor]:
    """Pack a flat token stream into non-overlapping ``seq_len + 1`` blocks."""

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    block_len = seq_len + 1
    buffer: list[int] = []
    for token in token_stream:
        buffer.append(int(token))
        if len(buffer) == block_len:
            yield torch.tensor(buffer, dtype=torch.long)
            buffer.clear()


class PackedDocumentIterator:
    """Stateful document-to-token block iterator.

    The iterator records how many source documents have been consumed and the
    current token buffer.  For HuggingFace streaming resumes, rebuild the
    underlying dataset with ``dataset.skip(documents_seen)`` and restore the
    buffer from checkpoint metadata.
    """

    def __init__(
        self,
        documents: Iterable[Any],
        encode: Callable[[str], Sequence[int]],
        *,
        seq_len: int,
        text_column: str = "text",
        documents_seen: int = 0,
        buffer_tokens: Sequence[int] | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if documents_seen < 0:
            raise ValueError("documents_seen must be non-negative")
        self._documents = iter(documents)
        self._encode = encode
        self.seq_len = seq_len
        self.text_column = text_column
        self.documents_seen = documents_seen
        self.buffer_tokens = [int(token) for token in (buffer_tokens or [])]
        self.blocks_emitted = 0

    def __iter__(self) -> "PackedDocumentIterator":
        return self

    def __next__(self) -> torch.Tensor:
        block_len = self.seq_len + 1
        while len(self.buffer_tokens) < block_len:
            document = next(self._documents)
            text = self._document_text(document)
            tokens = [int(token) for token in self._encode(text)]
            self.documents_seen += 1
            if tokens:
                self.buffer_tokens.extend(tokens)

        block = self.buffer_tokens[:block_len]
        del self.buffer_tokens[:block_len]
        self.blocks_emitted += 1
        return torch.tensor(block, dtype=torch.long)

    def _document_text(self, document: Any) -> str:
        if isinstance(document, dict):
            value = document.get(self.text_column, "")
        else:
            value = document
        return "" if value is None else str(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "seq_len": self.seq_len,
            "documents_seen": self.documents_seen,
            "buffer_tokens": list(self.buffer_tokens),
            "blocks_emitted": self.blocks_emitted,
        }


class FormulaTokenBlockIterator:
    """Deterministic infinite token source for smoke tests and unit tests."""

    def __init__(self, *, seq_len: int, vocab_size: int, seed: int = 0, cursor: int = 0) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.cursor = cursor

    def __iter__(self) -> "FormulaTokenBlockIterator":
        return self

    def __next__(self) -> torch.Tensor:
        block_len = self.seq_len + 1
        positions = torch.arange(self.cursor, self.cursor + block_len, dtype=torch.long)
        self.cursor += block_len
        tokens = (positions * 1_103_515_245 + self.seed) % self.vocab_size
        return tokens.to(dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "seed": self.seed,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cursor = int(state["cursor"])


class LMBlockBatchIterator:
    """Batch packed ``seq_len + 1`` blocks into causal-LM input/target pairs."""

    def __init__(self, block_iterator: Iterator[torch.Tensor], *, micro_batch_size: int) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        self.block_iterator = block_iterator
        self.micro_batch_size = micro_batch_size

    def __iter__(self) -> "LMBlockBatchIterator":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = [next(self.block_iterator) for _ in range(self.micro_batch_size)]
        packed = torch.stack(blocks, dim=0).to(dtype=torch.long)
        return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()

    def state_dict(self) -> dict[str, Any]:
        block_state = None
        if hasattr(self.block_iterator, "state_dict"):
            block_state = self.block_iterator.state_dict()  # type: ignore[attr-defined]
        return {
            "type": self.__class__.__name__,
            "micro_batch_size": self.micro_batch_size,
            "block_iterator": block_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        block_state = state.get("block_iterator")
        if block_state is not None and hasattr(self.block_iterator, "load_state_dict"):
            self.block_iterator.load_state_dict(block_state)  # type: ignore[attr-defined]


class FineWebPackedIterator:
    """Packed streaming iterator for FineWeb with held-out eval reservation."""

    def __init__(
        self,
        config: FineWebDataConfig,
        *,
        train: bool,
        documents_seen: int = 0,
        buffer_tokens: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        self.train = train
        self.documents_seen = documents_seen
        self.buffer_tokens = [int(token) for token in (buffer_tokens or [])]
        self.blocks_emitted = 0
        self._load_dataset, tokenizer_cls = _require_hf_data_deps()
        self._tokenizer = tokenizer_cls.from_pretrained(config.tokenizer_name)
        self._eos_token_id = self._tokenizer.eos_token_id
        self._documents = self._open_documents()

    def __iter__(self) -> "FineWebPackedIterator":
        return self

    def __next__(self) -> torch.Tensor:
        block_len = self.config.seq_len + 1
        while len(self.buffer_tokens) < block_len:
            document = next(self._documents)
            text = document.get(self.config.text_column, "") if isinstance(document, dict) else str(document)
            tokens = self._encode(text)
            if self.train:
                self.documents_seen += 1
            if tokens:
                self.buffer_tokens.extend(tokens)

        block = self.buffer_tokens[:block_len]
        del self.buffer_tokens[:block_len]
        self.blocks_emitted += 1
        return torch.tensor(block, dtype=torch.long)

    def _open_documents(self) -> Iterator[Any]:
        dataset = self._load_dataset(
            self.config.dataset_path,
            name=self.config.dataset_name,
            split=self.config.split,
            streaming=True,
        )
        if self.train:
            return iter(dataset.skip(self.config.eval_docs + self.documents_seen))
        return iter(dataset.take(self.config.eval_docs))

    def _encode(self, text: str) -> list[int]:
        tokens = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if self.config.add_eos and self._eos_token_id is not None:
            tokens = list(tokens) + [int(self._eos_token_id)]
        return [int(token) for token in tokens]

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "seq_len": self.config.seq_len,
            "train": self.train,
            "documents_seen": self.documents_seen,
            "data_skip_documents": self.documents_seen,
            "heldout_eval_docs": self.config.eval_docs,
            "buffer_tokens": list(self.buffer_tokens),
            "blocks_emitted": self.blocks_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.documents_seen = int(state.get("documents_seen", state.get("data_skip_documents", 0)))
        self.buffer_tokens = [int(token) for token in state.get("buffer_tokens", [])]
        self.blocks_emitted = int(state.get("blocks_emitted", 0))
        self._documents = self._open_documents()


def make_fineweb_train_iterator(
    config: FineWebDataConfig,
    *,
    micro_batch_size: int,
    state: dict[str, Any] | None = None,
) -> LMBlockBatchIterator:
    block_state = _extract_block_state(state)
    block_iterator = FineWebPackedIterator(
        config,
        train=True,
        documents_seen=int(block_state.get("documents_seen", 0)),
        buffer_tokens=block_state.get("buffer_tokens", []),
    )
    return LMBlockBatchIterator(block_iterator, micro_batch_size=micro_batch_size)


def make_fineweb_eval_iterator(config: FineWebDataConfig, *, micro_batch_size: int) -> LMBlockBatchIterator:
    block_iterator = FineWebPackedIterator(config, train=False)
    return LMBlockBatchIterator(block_iterator, micro_batch_size=micro_batch_size)


def make_synthetic_batch_iterator(
    *,
    seq_len: int,
    micro_batch_size: int,
    vocab_size: int,
    seed: int = 0,
    cursor: int = 0,
) -> LMBlockBatchIterator:
    block_iterator = FormulaTokenBlockIterator(seq_len=seq_len, vocab_size=vocab_size, seed=seed, cursor=cursor)
    return LMBlockBatchIterator(block_iterator, micro_batch_size=micro_batch_size)


def _extract_block_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    if state.get("type") == "LMBlockBatchIterator":
        return dict(state.get("block_iterator") or {})
    return state


def _require_hf_data_deps() -> tuple[Any, Any]:
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "FineWeb streaming requires datasets and transformers. "
            "Install them with: pip install -r train/requirements.txt"
        ) from exc
    return load_dataset, GPT2TokenizerFast


__all__ = [
    "FINEWEB_DATASET",
    "FINEWEB_NAME",
    "FINEWEB_SPLIT",
    "GPT2_TOKENIZER",
    "FineWebDataConfig",
    "FineWebPackedIterator",
    "FormulaTokenBlockIterator",
    "LMBlockBatchIterator",
    "PackedDocumentIterator",
    "make_fineweb_eval_iterator",
    "make_fineweb_train_iterator",
    "make_synthetic_batch_iterator",
    "pack_tokens",
]
