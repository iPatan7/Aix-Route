"""Local model implementations (Llama, Qwen, etc.)."""

from typing import Any

try:  # heavy optional dependency
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from deterministic_horizon.models.base import BaseModel, ModelResponse


class LocalModel(BaseModel):
    """Interface for locally-run models using transformers/vLLM."""

    # HuggingFace model IDs (paper's open-weight suite).
    MODEL_MAPPING = {
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    }

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_vllm: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize local model.

        Args:
            model_name: Model name or HuggingFace ID
            device: Device to run on ('auto', 'cuda', 'cpu')
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            use_vllm: Use vLLM for faster inference
            **kwargs: Additional arguments
        """
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_vllm = use_vllm

        super().__init__(model_name=model_name, **kwargs)

    def _get_model_id(self) -> str:
        """Get HuggingFace model ID."""
        model_lower = self.model_name.lower()
        for key, model_id in self.MODEL_MAPPING.items():
            if key in model_lower:
                return model_id
        return self.model_name

    def _setup_client(self) -> None:
        """Set up local model."""
        if self.use_vllm:
            self._setup_vllm()
        else:
            self._setup_transformers()

    def _setup_vllm(self) -> None:
        """Set up vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as err:
            raise ImportError("vllm package required. Install with: pip install vllm") from err

        model_id = self._get_model_id()

        self._client = LLM(
            model=model_id,
            dtype="auto",
            trust_remote_code=True,
            max_model_len=8192,
        )
        self._sampling_params_class = SamplingParams

    def _setup_transformers(self) -> None:
        """Set up transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as err:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            ) from err

        model_id = self._get_model_id()

        # Quantization config
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Load model
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self._client = (self._model, self._tokenizer)

    def _call_api(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate response from local model."""
        if self.use_vllm:
            return self._call_vllm(messages, **kwargs)
        else:
            return self._call_transformers(messages, **kwargs)

    def _call_vllm(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """Generate with vLLM."""
        # Format messages into prompt
        prompt = self._format_messages(messages)

        sampling_params = self._sampling_params_class(
            temperature=self.temperature if self.temperature > 0 else 0.01,
            max_tokens=self.max_tokens,
            top_p=0.95 if self.temperature > 0 else 1.0,
        )

        outputs = self._client.generate([prompt], sampling_params)
        output = outputs[0]

        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        }

    def _call_transformers(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """Generate with transformers."""
        model, tokenizer = self._client

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_tokens = outputs[0][prompt_tokens:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(generated_tokens),
        }

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages for models without chat template support."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    def _parse_response(
        self,
        raw_response: dict[str, Any],
        latency_ms: float,
    ) -> ModelResponse:
        """Parse local model response."""
        return ModelResponse(
            content=raw_response.get("text", ""),
            model=self.model_name,
            prompt_tokens=raw_response.get("prompt_tokens", 0),
            completion_tokens=raw_response.get("completion_tokens", 0),
            total_tokens=(
                raw_response.get("prompt_tokens", 0) + raw_response.get("completion_tokens", 0)
            ),
            latency_ms=latency_ms,
            raw_response=raw_response,
        )


class LlamaModel(LocalModel):
    """Llama-specific implementation.

    The paper's open-weight Llama suite is Llama-3.1-8B and Llama-3.3-70B, so
    the family version depends on size: 8B → Llama-3.1, 70B → Llama-3.3.
    """

    _SIZE_TO_NAME = {"8b": "llama-3.1-8b", "70b": "llama-3.3-70b"}

    def __init__(self, size: str = "8b", **kwargs) -> None:
        model_name = self._SIZE_TO_NAME.get(size.lower(), f"llama-3.1-{size}")
        super().__init__(model_name=model_name, **kwargs)


class QwenModel(LocalModel):
    """Qwen-specific implementation."""

    def __init__(self, size: str = "7b", **kwargs) -> None:
        model_name = f"qwen-2.5-{size}"
        super().__init__(model_name=model_name, **kwargs)
