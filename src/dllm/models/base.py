"""Public model-adapter contract used by inference and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    block_size: int = 32
    steps: int | None = None
    threshold: float = 0.9
    temperature: float = 0.0
    seed: int = 42
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.steps is not None and self.steps <= 0:
            raise ValueError("steps must be positive")


@dataclass
class GenerationResult:
    prompt: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    text: str
    elapsed_seconds: float
    model_evaluations: int | None = None
    termination_reason: str = "length"
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """One model family's loading, tokenization, and native decoding policy."""

    name: str
    attention_class_names: tuple[str, ...] = ()
    attention_integration: str = "registry"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        precision: str = "bfloat16",
        revision: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.revision = revision
        self.model: Any = None
        self.tokenizer: Any = None

    @abstractmethod
    def load(self) -> "ModelAdapter":
        """Load the model and tokenizer and return this adapter."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run the model's native dLLM decoder."""

    def load_tokenizer(self) -> "ModelAdapter":
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            revision=self.revision,
            trust_remote_code=True,
        )
        return self

    def encode_prompt(
        self, prompt: str, extra: Mapping[str, Any] | None = None
    ) -> list[int]:
        encoded = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return list(encoded)

    def prompt_configuration(
        self, extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return adapter-specific options that change prompt tokenization."""
        return {}

    def is_blasst_attention_module(self, name: str, module: Any) -> bool:
        """Select attention modules supported by this adapter's BLASST binding."""
        return type(module).__name__ in self.attention_class_names

    def blasst_query_ids(self, module: Any, args: tuple, kwargs: dict) -> Any:
        """Extract the IDs whose rows correspond to the active attention queries."""
        value = kwargs.get("input_ids")
        if value is None and args:
            value = args[0]
        return value

    def blasst_dense_kv_prefix(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        kwargs: Mapping[str, Any],
    ) -> int:
        """Return a leading KV length that must never be sparsified."""
        return 0

    def blasst_call_is_eligible(
        self, module: Any, args: tuple, kwargs: Mapping[str, Any]
    ) -> bool:
        """Return whether the current model forward should use BLASST/eager attention."""
        return True

    @property
    def blasst_filter_special_query_ids(self) -> bool:
        return True

    def runtime_metadata(self) -> dict[str, Any]:
        """Dependency and resolved-checkpoint metadata available after loading."""
        result: dict[str, Any] = {}
        try:
            import transformers

            result["transformers_version"] = transformers.__version__
        except ImportError:
            pass
        config = getattr(self.model, "config", None)
        resolved = getattr(config, "_commit_hash", None)
        if resolved:
            result["resolved_model_revision"] = str(resolved)
        result["requested_model_revision"] = self.revision
        return result

    def tokenizer_identity(self) -> str:
        return str(getattr(self.tokenizer, "name_or_path", self.model_path))

    @property
    def mask_token_id(self) -> int | None:
        value = getattr(self.tokenizer, "mask_token_id", None)
        if value is None and self.model is not None:
            value = getattr(self.model.config, "mask_token_id", None)
        return int(value) if value is not None else None

    @property
    def pad_token_id(self) -> int | None:
        value = getattr(self.tokenizer, "pad_token_id", None)
        return int(value) if value is not None else None

    def capability(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "attention_class_names": list(self.attention_class_names),
            "attention_integration": self.attention_integration,
            "mask_token_id": self.mask_token_id,
            "pad_token_id": self.pad_token_id,
        }
