import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# GPT-2 / Llama-style split pattern. Isolates contractions, words, numbers,
# punctuation runs, and whitespace before byte-level encoding.
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Phrases the trained tokenizer must round-trip exactly (no strip).
# Cover ASCII, leading/trailing whitespace, unicode, CJK, emoji, code, numbers.
ROUND_TRIP_TESTS = [
    "Hello world! This is a validation check, 1234.",
    "  leading and trailing spaces  ",
    "naïve café — résumé Æthelred",
    "中文测试 한국어 テスト",
    "emoji: 😀🔥👨‍👩‍👧",
    "code: def f(x): return x**2  # comment",
    "numbers 3.14159 1,000,000 -42 1e-9",
    "\n\nnewlines\tand\ttabs\n",
]


@dataclass
class TokenizerConfig:
    vocab_size: int
    special_tokens: List[str]
    batch_size: int = 1000
    min_frequency: int = 2
    text_field: str = "text"
    model_max_length: int = 8192
    unk_token: Optional[str] = "<unk>"
    pad_token: Optional[str] = "<pad>"
    bos_token: Optional[str] = "<bos>"
    eos_token: Optional[str] = "<eos>"
    additional_special_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.special_tokens:
            raise ValueError("special_tokens must be non-empty")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens contains duplicates")
        for t in self.special_tokens:
            if not t or t.strip() != t:
                raise ValueError(f"invalid special token: {t!r}")
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed "
                f"len(special_tokens) ({len(self.special_tokens)}) to allow merges"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        # Required HF special tokens must be in special_tokens
        for name in ("unk_token", "pad_token", "bos_token", "eos_token"):
            tok = getattr(self, name)
            if tok is not None and tok not in self.special_tokens:
                raise ValueError(
                    f"{name}={tok!r} not in special_tokens; "
                    f"either add it or set {name}=None"
                )


class TokenizerTrainer:
    def __init__(
        self,
        dataset: Iterable[Any],
        config: TokenizerConfig,
        text_extractor: Optional[Callable[[Any], str]] = None,
    ):
        self.dataset = dataset
        self.config = config
        self.text_extractor = text_extractor or self._default_extractor
        self.tokenizer: Optional[Tokenizer] = None

    def _default_extractor(self, sample: Any) -> str:
        if isinstance(sample, dict):
            return str(sample.get(self.config.text_field, "")).strip()
        return str(sample).strip()

    def _build_tokenizer(self) -> Tokenizer:
        tok = Tokenizer(BPE(unk_token=self.config.unk_token))
        tok.pre_tokenizer = Sequence([
            Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = ByteLevelDecoder()
        return tok

    def _build_trainer(self) -> BpeTrainer:
        return BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )

    def _batches(self) -> Iterable[List[str]]:
        buf, n_yielded = [], 0
        for sample in self.dataset:
            text = self.text_extractor(sample)
            if not text:
                continue
            buf.append(text)
            if len(buf) >= self.config.batch_size:
                yield buf
                n_yielded += len(buf)
                if n_yielded % (self.config.batch_size * 100) == 0:
                    logger.info("processed %d texts", n_yielded)
                buf = []
        if buf:
            yield buf

    def _attach_post_processor(self) -> None:
        """Look up real IDs from the trained tokenizer, not input order."""
        bos, eos = self.config.bos_token, self.config.eos_token
        if not (bos and eos):
            logger.info("bos/eos not configured; skipping post-processor")
            return
        bos_id = self.tokenizer.token_to_id(bos)
        eos_id = self.tokenizer.token_to_id(eos)
        if bos_id is None or eos_id is None:
            raise RuntimeError(
                f"bos/eos missing from trained vocab: "
                f"{bos}={bos_id}, {eos}={eos_id}"
            )
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{bos} $A {eos}",
            pair=f"{bos} $A {eos} {bos} $B:1 {eos}:1",
            special_tokens=[(bos, bos_id), (eos, eos_id)],
        )

    def _verify_round_trip(self) -> None:
        """Strict round-trip: no strip, exact match required."""
        failures = []
        for phrase in ROUND_TRIP_TESTS:
            try:
                ids = self.tokenizer.encode(phrase, add_special_tokens=False).ids
                decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                if decoded != phrase:
                    failures.append((phrase, decoded))
            except Exception as e:
                failures.append((phrase, f"<exception: {e}>"))
        if failures:
            for orig, got in failures:
                logger.error("round-trip FAIL\n  in:  %r\n  out: %r", orig, got)
            raise RuntimeError(f"{len(failures)} round-trip failures")
        logger.info("round-trip OK on %d phrases", len(ROUND_TRIP_TESTS))

    def _verify_special_tokens(self) -> None:
        for t in self.config.special_tokens:
            if self.tokenizer.token_to_id(t) is None:
                raise RuntimeError(f"special token {t!r} missing from vocab")

    def _save_hf(self, raw_tokenizer_path: str, save_dir: str) -> None:
        kwargs = {
            "tokenizer_file": raw_tokenizer_path,
            "model_max_length": self.config.model_max_length,
            "additional_special_tokens": self.config.additional_special_tokens,
        }
        for name in ("unk_token", "pad_token", "bos_token", "eos_token"):
            val = getattr(self.config, name)
            if val is not None:
                kwargs[name] = val
        hf = PreTrainedTokenizerFast(**kwargs)
        hf.save_pretrained(save_dir)

    def _write_metadata(self, save_dir: str) -> None:
        meta = {
            "vocab_size_target": self.config.vocab_size,
            "vocab_size_actual": self.tokenizer.get_vocab_size(),
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "split_pattern": SPLIT_PATTERN,
        }
        with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def train(self, output_dir: str) -> Tokenizer:
        # Pre-flight: make sure target is writable before doing the work.
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"output_dir not writable: {output_dir}")

        self.tokenizer = self._build_tokenizer()
        trainer = self._build_trainer()

        logger.info("training: target_vocab=%d", self.config.vocab_size)
        self.tokenizer.train_from_iterator(self._batches(), trainer=trainer)
        logger.info("training done: actual_vocab=%d",
                    self.tokenizer.get_vocab_size())

        self._attach_post_processor()
        self._verify_special_tokens()
        self._verify_round_trip()

        # Atomic save: build everything in a tmp dir, then swap in.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(output_dir) or ".") as tmp:
            raw_path = os.path.join(tmp, "tokenizer.json")
            self.tokenizer.save(raw_path)
            self._save_hf(raw_path, tmp)
            self._write_metadata(tmp)

            # Replace contents of output_dir atomically-ish
            for name in os.listdir(tmp):
                src = os.path.join(tmp, name)
                dst = os.path.join(output_dir, name)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)

        logger.info("saved to %s", output_dir)
        return self.tokenizer