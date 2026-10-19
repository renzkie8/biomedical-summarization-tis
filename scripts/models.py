from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    max_source_length: int
    max_target_length: int = 256
    use_global_attention: bool = False
    parameters: str = "406M"
    attention_type: str = "Dense Self-Attention"
    role: str = "Baseline"
    complexity: str = "O(n²)"
    mechanism_details: str = "Full dense self-attention"
    initialization: str = "HuggingFace pretrained"

BART_SPEC = ModelSpec(
    name="facebook/bart-large-cnn", 
    max_source_length=1024, 
    max_target_length=256,
    role="Truncation-constrained baseline"
)
LED_SPEC = ModelSpec(
    name="allenai/led-large-16384", 
    max_source_length=4096, 
    max_target_length=256, 
    use_global_attention=True,
    attention_type="Sparse Local/Global Attention",
    complexity="O(n × w)",
    mechanism_details="Sliding-window local + global (BOS)",
    initialization="Extended from BART-large weights",
    role="Long-context summariser"
)


class HuggingFaceSummarizer:
    def __init__(self, model_name: str, *, max_source_length: int = 1024, max_target_length: int = 256, device: str | None = None):
        self.model_name = model_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.device = device
        # Annotate as Any to satisfy static analyzers that can't resolve
        # dynamic imports from `transformers` and `torch` at analysis time.
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._use_fp16 = False
        self._logger = logging.getLogger("tis_eval")

    @staticmethod
    def _has_model_weights(model_dir: Path) -> bool:
        """Return True when a local directory contains usable HF model weights."""
        if not model_dir.exists() or not model_dir.is_dir():
            return False

        weight_files = (
            "model.safetensors",
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        return any((model_dir / filename).exists() for filename in weight_files)

    def _resolve_model_source(self) -> str:
        """Choose a loadable model source, preferring local weights when present."""
        candidate_path = Path(self.model_name)
        if candidate_path.exists() and self._has_model_weights(candidate_path):
            return str(candidate_path)

        lowered_name = self.model_name.lower()
        if "bart" in lowered_name:
            return "facebook/bart-large-cnn"
        if "led" in lowered_name:
            return "allenai/led-large-16384"
        return self.model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            transformers = importlib.import_module("transformers")
            AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
            AutoTokenizer = transformers.AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Hugging Face summarization requires `torch` and `transformers`. "
                "Install the project dependencies before calling this backend."
            ) from exc

        self._torch = torch

        # If loading from a local LoRA or output directory, the tokenizer/model
        # files might not be present. Prefer local weights when available and fall
        # back to the base pretrained checkpoint otherwise.
        model_source = self._resolve_model_source()
        if model_source == "facebook/bart-large-cnn":
            tokenizer_name = "facebook/bart-large-cnn"
        elif model_source == "allenai/led-large-16384":
            tokenizer_name = "allenai/led-large-16384"
        else:
            tokenizer_name = model_source

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._logger.info("Loading summarizer from %s", model_source)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)
        self._model.to(self._device)
            
        self._model.eval()

        # Decide whether to use FP16 (only on CUDA)
        try:
            self._use_fp16 = (self._device.type == "cuda") and torch.cuda.is_available()
            if self._use_fp16:
                try:
                    # Convert model parameters to fp16 to reduce memory and speed up inference
                    self._model.half()
                    self._logger.info("Enabled model.half() for fp16 inference")
                except Exception as e:
                    self._use_fp16 = False
                    self._logger.warning("Could not enable fp16 (model.half() failed): %s; continuing with fp32", e)
        except Exception:
            self._use_fp16 = False

        # Optional: attempt torch.compile if explicitly enabled via env var and supported
        try:
            enable_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if enable_compile and torch_version >= (2, 0):
                try:
                    self._model = torch.compile(self._model)
                    self._logger.info("torch.compile applied to model")
                except Exception as e:
                    self._logger.warning("torch.compile failed, continuing without compile: %s", e)
        except Exception:
            # If anything goes wrong determining versions or compile, continue safely
            pass

    def generate(self, article: str, *, num_beams: int = 4, min_length: int = 32) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        encoded = self._tokenizer(
            article,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        if "led" in self.model_name.lower():
            global_attention_mask = self._torch.zeros_like(encoded["input_ids"])
            global_attention_mask[:, 0] = 1
            encoded["global_attention_mask"] = global_attention_mask
        # Use inference_mode for faster, memory-efficient inference. If fp16 is
        # enabled and we're on CUDA, use autocast to accelerate kernels safely.
        with self._torch.inference_mode():
            if self._use_fp16 and getattr(self._device, "type", "cpu") == "cuda":
                autocast_ctx = getattr(self._torch.cuda.amp, "autocast")
                with autocast_ctx():
                    output_ids = self._model.generate(
                        **encoded,
                        num_beams=num_beams,
                        max_length=self.max_target_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
            else:
                output_ids = self._model.generate(
                    **encoded,
                    num_beams=num_beams,
                    max_length=self.max_target_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def generate_many(self, articles: Iterable[str]) -> list[str]:
        return [self.generate(article) for article in articles]

    def generate_batch(self, articles: list[str], *, batch_size: int = 4, num_beams: int = 4, min_length: int = 32, progress_bar=None) -> list[str]:
        """Generate summaries for multiple articles using batch processing for efficiency."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        
        results = []
        num_batches = (len(articles) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(articles))
            batch_articles = articles[start_idx:end_idx]
            
            # Tokenize batch
            encoded = self._tokenizer(
                batch_articles,
                max_length=self.max_source_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            
            # Add global attention mask for LED
            if "led" in self.model_name.lower():
                global_attention_mask = self._torch.zeros_like(encoded["input_ids"])
                global_attention_mask[:, 0] = 1
                encoded["global_attention_mask"] = global_attention_mask
            
            # Generate batch under inference mode; use autocast on CUDA fp16 when enabled
            with self._torch.inference_mode():
                if self._use_fp16 and getattr(self._device, "type", "cpu") == "cuda":
                    autocast_ctx = getattr(self._torch.cuda.amp, "autocast")
                    with autocast_ctx():
                        output_ids = self._model.generate(
                            **encoded,
                            num_beams=num_beams,
                            max_length=self.max_target_length,
                            min_length=min_length,
                            length_penalty=2.0,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                        )
                else:
                    output_ids = self._model.generate(
                        **encoded,
                        num_beams=num_beams,
                        max_length=self.max_target_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
            
            # Decode batch
            batch_results = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            results.extend(batch_results)
            
            # Update progress bar if provided
            if progress_bar is not None:
                progress_bar.update(len(batch_articles))
        
        return results
