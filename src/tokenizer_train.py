import json
import logging
import os
import shutil
import tempfile
from typing import Any, Callable, Iterable, List, Optional

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from configs.tokenizer_config import TokenizerConfig

logger = logging.getLogger(__name__)

SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



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



class TokenizerTrainer:
    def __init__(
        self,
        dataset: Iterable[Any],
        config: TokenizerConfig,
        text_extractor: Optional[Callable[[Any], str]] = None,
    ):
        """Initialize the trainer with a dataset, config, and optional text extractor.

        Args:
            dataset: Iterable of samples to train the tokenizer on.
            config: Tokenizer configuration (vocab size, special tokens, etc.).
            text_extractor: Optional callable that maps a sample to its text.
                Defaults to :meth:`_default_extractor` when not provided.
        """
        if dataset is None:
            raise ValueError("dataset must not be None")
        if config is None:
            raise ValueError("config must not be None")
        if text_extractor is not None and not callable(text_extractor):
            raise TypeError("text_extractor must be callable")
        self.dataset = dataset
        self.config = config
        self.text_extractor = text_extractor or self._default_extractor
        self.tokenizer: Optional[Tokenizer] = None

    def _default_extractor(self, sample: Any) -> str:
        """Extract text from a sample.

        For dict samples, reads the configured ``text_field``; otherwise casts
        the sample to a string. The result is stripped of surrounding whitespace.

        Args:
            sample: A single dataset sample.

        Returns:
            The extracted, whitespace-stripped text.
        """
        try:
            if isinstance(sample, dict):
                return str(sample.get(self.config.text_field, "")).strip()
            return str(sample).strip()
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("failed to extract text from sample: %s", e)
            return ""

    def _build_tokenizer(self) -> Tokenizer:
        """Construct an untrained byte-level BPE tokenizer.

        Wires up a GPT-style regex split pre-tokenizer followed by byte-level
        encoding, and a matching byte-level decoder for reversible round trips.

        Returns:
            A configured but untrained :class:`Tokenizer` instance.
        """
        logger.debug("building byte-level BPE tokenizer (unk_token=%r)",
                     self.config.unk_token)
        tok = Tokenizer(BPE(unk_token=self.config.unk_token))
        tok.pre_tokenizer = Sequence([
            Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = ByteLevelDecoder()
        return tok

    def _build_trainer(self) -> BpeTrainer:
        """Create the BPE trainer from the configuration.

        Seeds the initial alphabet with the full byte-level alphabet so every
        byte is representable and training stays reversible.

        Returns:
            A configured :class:`BpeTrainer` instance.
        """
        logger.debug(
            "building BPE trainer (vocab_size=%d, min_frequency=%d, special_tokens=%d)",
            self.config.vocab_size,
            self.config.min_frequency,
            len(self.config.special_tokens),
        )
        return BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )

    def _batches(self) -> Iterable[List[str]]:
        """Yield batches of extracted text for streaming training.

        Iterates the dataset, extracts and skips empty text, and yields lists of
        up to ``batch_size`` strings, logging progress periodically. A final
        partial batch is yielded if any leftover texts remain.

        Yields:
            Lists of non-empty text strings.
        """
        buf, n_yielded, n_skipped = [], 0, 0
        for sample in self.dataset:
            try:
                text = self.text_extractor(sample)
            except Exception as e:
                # text_extractor is arbitrary/user-supplied; no specific
                # exception type to catch. Skip the bad sample rather than
                # aborting a long training run.
                n_skipped += 1
                logger.warning("text_extractor raised, skipping sample: %s", e)
                continue
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
        if n_skipped:
            logger.warning("skipped %d samples due to extraction errors", n_skipped)

    def _attach_post_processor(self) -> None:
        """Attach a BOS/EOS template post-processor to the trained tokenizer.

        Resolves the actual token IDs from the trained vocab (rather than
        assuming input order) and installs a :class:`TemplateProcessing` that
        wraps single and paired sequences with BOS/EOS. No-op when BOS/EOS are
        not configured.

        Raises:
            RuntimeError: If a configured BOS/EOS token is absent from the vocab.
        """
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
        logger.info("attached post-processor (bos=%s:%d, eos=%s:%d)",
                    bos, bos_id, eos, eos_id)

    def _verify_round_trip(self) -> None:
        """Verify encode/decode is lossless on a set of edge-case phrases.

        Encodes then decodes each phrase in :data:`ROUND_TRIP_TESTS` and requires
        an exact match (no stripping). Aggregates and logs all mismatches or
        exceptions before failing.

        Raises:
            RuntimeError: If any phrase fails to round-trip exactly.
        """
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
        """Ensure every configured special token exists in the trained vocab.

        Raises:
            RuntimeError: If any configured special token is missing.
        """
        for t in self.config.special_tokens:
            if self.tokenizer.token_to_id(t) is None:
                raise RuntimeError(f"special token {t!r} missing from vocab")
        logger.info("verified %d special tokens present in vocab",
                    len(self.config.special_tokens))

    def _save_hf(self, raw_tokenizer_path: str, save_dir: str) -> None:
        """Wrap the raw tokenizer as a HuggingFace tokenizer and save it.

        Builds a :class:`PreTrainedTokenizerFast` from the raw tokenizer file,
        carrying over max length, special tokens, and any configured
        unk/pad/bos/eos tokens, then writes it to ``save_dir``.

        Args:
            raw_tokenizer_path: Path to the saved raw ``tokenizer.json``.
            save_dir: Directory to write the HuggingFace tokenizer files into.
        """
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
        logger.debug("saved HuggingFace tokenizer to %s", save_dir)

    def _write_metadata(self, save_dir: str) -> None:
        """Write training metadata as JSON alongside the tokenizer.

        Records target vs. actual vocab size, min frequency, special tokens, and
        the split pattern to ``training_metadata.json`` in ``save_dir``.

        Args:
            save_dir: Directory to write the metadata file into.
        """
        meta = {
            "vocab_size_target": self.config.vocab_size,
            "vocab_size_actual": self.tokenizer.get_vocab_size(),
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "split_pattern": SPLIT_PATTERN,
        }
        try:
            with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except OSError as e:
            raise RuntimeError(f"failed to write training metadata: {e}") from e
        logger.debug("wrote training metadata to %s", save_dir)

    def train(self, output_dir: str) -> Tokenizer:
        """Train the tokenizer end to end and save it to ``output_dir``.

        Builds the tokenizer and trainer, trains from the batched dataset,
        attaches the post-processor, verifies special tokens and round trips,
        then atomically writes the raw tokenizer, HuggingFace wrapper, and
        metadata into ``output_dir`` via a temporary staging directory.

        Args:
            output_dir: Directory to save the trained tokenizer artifacts into.

        Returns:
            The trained :class:`Tokenizer` instance.

        Raises:
            PermissionError: If ``output_dir`` is not writable.
            RuntimeError: If training fails, produces an empty vocab, verification
                of special tokens or round trips fails, or artifacts cannot be
                saved.
        """
        logger.info("starting tokenizer training run -> %s", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"output_dir not writable: {output_dir}")

        self.tokenizer = self._build_tokenizer()
        trainer = self._build_trainer()

        logger.info("training: target_vocab=%d", self.config.vocab_size)
        try:
            self.tokenizer.train_from_iterator(self._batches(), trainer=trainer)
        except Exception as e:
            raise RuntimeError(f"tokenizer training failed: {e}") from e

        actual_vocab = self.tokenizer.get_vocab_size()
        if actual_vocab <= len(self.config.special_tokens):
            raise RuntimeError(
                f"training produced an empty vocab (size={actual_vocab}); "
                f"the dataset may be empty or yielded no usable text"
            )
        logger.info("training done: actual_vocab=%d", actual_vocab)

        self._attach_post_processor()
        self._verify_special_tokens()
        self._verify_round_trip()

        logger.info("saving tokenizer artifacts to %s", output_dir)
        try:
            with tempfile.TemporaryDirectory(dir=os.path.dirname(output_dir) or ".") as tmp:
                logger.debug("staging artifacts in temp dir %s", tmp)
                raw_path = os.path.join(tmp, "tokenizer.json")
                self.tokenizer.save(raw_path)
                self._save_hf(raw_path, tmp)
                self._write_metadata(tmp)

                for name in os.listdir(tmp):
                    src = os.path.join(tmp, name)
                    dst = os.path.join(output_dir, name)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)
                    logger.debug("moved artifact %s -> %s", name, dst)
        except OSError as e:
            raise RuntimeError(
                f"failed to save tokenizer artifacts to {output_dir}: {e}"
            ) from e

        logger.info("saved to %s", output_dir)
        return self.tokenizer


