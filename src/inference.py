"""
Inference Engine

Production-ready inference server with:
- FastAPI REST API
- Batched generation
- Streaming responses
- Model caching
- Prometheus metrics
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

import torch
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    Request = None
    CORSMiddleware = None
    Response = None
    StreamingResponse = None

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = MagicMock if 'MagicMock' in globals() else None
    generate_latest = lambda: b""
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

from src.config import InferenceConfig, GenerationConfig as ConfigGenerationConfig
from src.model_utils import load_model_and_tokenizer, merge_and_unload_peft
from src.logger import get_logger

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelInstance:
    """Wrapper for model and tokenizer with metadata."""
    model: torch.nn.Module
    tokenizer: Any
    config: InferenceConfig
    status: ModelStatus = ModelStatus.UNLOADED
    loaded_at: Optional[float] = None
    request_count: int = 0
    total_tokens_generated: int = 0
    last_used: float = 0.0


class InferenceEngine:
    """Main inference engine managing model lifecycle and generation."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model_instance: Optional[ModelInstance] = None
        self._generation_lock = asyncio.Lock()
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Prometheus metrics
        self._setup_metrics()

    def _setup_metrics(self):
        if not PROMETHEUS_AVAILABLE:
            return
        self.requests_total = Counter(
            "inference_requests_total",
            "Total inference requests",
            ["model", "status"]
        )
        self.request_duration = Histogram(
            "inference_request_duration_seconds",
            "Request duration",
            ["model", "endpoint"]
        )
        self.tokens_generated = Counter(
            "inference_tokens_generated_total",
            "Total tokens generated",
            ["model"]
        )
        self.active_requests = Gauge(
            "inference_active_requests",
            "Currently processing requests",
            ["model"]
        )
        self.queue_size = Gauge(
            "inference_queue_size",
            "Request queue size",
            ["model"]
        )
        self.gpu_memory = Gauge(
            "inference_gpu_memory_gb",
            "GPU memory usage",
            ["model", "device"]
        )

    async def load_model(self, config: Optional[InferenceConfig] = None) -> ModelInstance:
        """Load model and tokenizer."""
        config = config or self.config
        logger.info(f"Loading model: {config.model_path}")

        self.model_instance = ModelInstance(
            model=None,
            tokenizer=None,
            config=config,
            status=ModelStatus.LOADING,
        )

        try:
            model, tokenizer = load_model_and_tokenizer(
                model_name_or_path=config.model_path,
                tokenizer_name_or_path=config.tokenizer_path,
                quantization=config.quantization,
                device_map=config.device_map,
                torch_dtype=config.torch_dtype,
                trust_remote_code=config.trust_remote_code,
            )

            # Apply optimizations
            if config.torch_compile:
                model = torch.compile(model, mode="reduce-overhead")

            if config.use_flash_attention:
                model.config._attn_implementation = "flash_attention_2"

            model.eval()

            self.model_instance.model = model
            self.model_instance.tokenizer = tokenizer
            self.model_instance.status = ModelStatus.READY
            self.model_instance.loaded_at = time.time()

            logger.info(f"Model loaded successfully in {time.time() - self.model_instance.loaded_at:.2f}s")
            return self.model_instance

        except Exception as e:
            self.model_instance.status = ModelStatus.ERROR
            logger.error(f"Failed to load model: {e}")
            raise

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model_instance:
            del self.model_instance.model
            del self.model_instance.tokenizer
            torch.cuda.empty_cache()
            self.model_instance.status = ModelStatus.UNLOADED
            self.model_instance = None
            logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        generation_config: Optional[ConfigGenerationConfig] = None,
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from prompt."""
        if not self.model_instance or self.model_instance.status != ModelStatus.READY:
            raise RuntimeError("Model not loaded or not ready")

        async with self._generation_lock:
            self.model_instance.request_count += 1
            self.model_instance.last_used = time.time()
            self.active_requests.labels(model=self.config.model_path).inc()

            try:
                gen_config = generation_config or self._get_default_gen_config()
                inputs = self.model_instance.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_input_length,
                ).to(self.model_instance.model.device)

                start_time = time.time()
                input_length = inputs.input_ids.shape[1]

                if stream:
                    return self._stream_generate(inputs, gen_config, start_time, input_length)
                else:
                    return await self._generate_complete(inputs, gen_config, start_time, input_length)

            except Exception as e:
                self.requests_total.labels(
                    model=self.config.model_path,
                    status="error"
                ).inc()
                raise
            finally:
                self.active_requests.labels(model=self.config.model_path).dec()

    async def _generate_complete(
        self,
        inputs: Dict[str, torch.Tensor],
        gen_config: ConfigGenerationConfig,
        start_time: float,
        input_length: int,
    ) -> str:
        """Generate complete response."""
        with torch.no_grad():
            outputs = self.model_instance.model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=self.model_instance.tokenizer.pad_token_id,
                eos_token_id=self.model_instance.tokenizer.eos_token_id,
            )

        generated = outputs[0][input_length:]
        response = self.model_instance.tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        self._record_metrics(start_time, len(generated))
        return response.strip()

    async def _stream_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        gen_config: ConfigGenerationConfig,
        start_time: float,
        input_length: int,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens."""
        streamer = TextIteratorStreamer(
            self.model_instance.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            generation_config=gen_config,
            streamer=streamer,
            pad_token_id=self.model_instance.tokenizer.pad_token_id,
            eos_token_id=self.model_instance.tokenizer.eos_token_id,
        )

        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(
            None,
            lambda: self.model_instance.model.generate(**generation_kwargs)
        )

        total_tokens = 0
        async for token in self._async_streamer(streamer):
            total_tokens += 1
            yield token

        await task
        self._record_metrics(start_time, total_tokens)

    async def _async_streamer(self, streamer: TextIteratorStreamer) -> AsyncGenerator[str, None]:
        """Convert sync streamer to async generator."""
        for token in streamer:
            yield token
            await asyncio.sleep(0)  # Yield control

    def _get_default_gen_config(self) -> ConfigGenerationConfig:
        """Get default generation config from model config."""
        return ConfigGenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            early_stopping=True,
        )

    def _record_metrics(self, start_time: float, tokens_generated: int) -> None:
        """Record generation metrics."""
        duration = time.time() - start_time
        self.request_duration.labels(
            model=self.config.model_path,
            endpoint="generate"
        ).observe(duration)
        self.tokens_generated.labels(model=self.config.model_path).inc(tokens_generated)
        self.requests_total.labels(
            model=self.config.model_path,
            status="success"
        ).inc()

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        if not self.model_instance:
            return {"status": "unloaded"}

        return {
            "status": self.model_instance.status.value,
            "model_path": self.config.model_path,
            "loaded_at": self.model_instance.loaded_at,
            "request_count": self.model_instance.request_count,
            "total_tokens_generated": self.model_instance.total_tokens_generated,
            "last_used": self.model_instance.last_used,
            "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
        }


class ChatCompletionEngine(InferenceEngine):
    """Engine for OpenAI-compatible chat completions."""

    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self._chat_templates = {}

    def register_chat_template(self, name: str, template: str) -> None:
        """Register a chat template."""
        self._chat_templates[name] = template

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        template_name: str = "default",
    ) -> str:
        """Apply chat template to messages."""
        if template_name in self._chat_templates:
            template = self._chat_templates[template_name]
        else:
            template = self._get_default_template()

        return self.model_instance.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _get_default_template(self) -> str:
        """Get default chat template."""
        return (
            "{% for message in messages %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
            "assistant: "
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[ConfigGenerationConfig] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """OpenAI-compatible chat completion."""
        prompt = self.apply_chat_template(messages)

        if stream:
            return self._stream_chat_completion(prompt, generation_config)
        else:
            return await self._complete_chat_completion(prompt, generation_config)

    async def _complete_chat_completion(
        self,
        prompt: str,
        generation_config: Optional[ConfigGenerationConfig],
    ) -> Dict[str, Any]:
        response = await self.generate(prompt, generation_config, stream=False)
        
        prompt_tokens = 0
        completion_tokens = 0
        if self.model_instance and self.model_instance.tokenizer:
            try:
                prompt_tokens = len(self.model_instance.tokenizer.encode(prompt))
                completion_tokens = len(self.model_instance.tokenizer.encode(response))
            except Exception:
                prompt_tokens = len(prompt.split())
                completion_tokens = len(response.split())
        else:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response.split())
        
        total_tokens = prompt_tokens + completion_tokens

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    async def _stream_chat_completion(
        self,
        prompt: str,
        generation_config: Optional[ConfigGenerationConfig],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion chunks."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Initial chunk
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_path,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        }

        async for token in self.generate(prompt, generation_config, stream=True):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.model_path,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }

        # Final chunk
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_path,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }


class InferenceServer:
    """FastAPI inference server."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = ChatCompletionEngine(config)
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="LLM Inference Server",
            version="1.0.0",
            lifespan=self._lifespan,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model": self.engine.get_status(),
            }

        @app.get("/metrics")
        async def metrics():
            return Response(content=generate_latest(), media_type="text/plain")

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": self.config.model_path,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "user",
                }],
            }

        @app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            return await self._handle_completion(request)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat_completion(request)

        @app.post("/v1/generate")
        async def generate(request: GenerateRequest):
            return await self._handle_generate(request)

        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        logger.info("Starting inference server...")
        await self.engine.load_model()
        yield
        logger.info("Shutting down inference server...")
        await self.engine.unload_model()

    async def _handle_completion(self, request: CompletionRequest) -> Union[Dict, StreamingResponse]:
        gen_config = ConfigGenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            num_beams=1,
        )

        if request.stream:
            async def stream_generator():
                async for token in self.engine.generate(
                    request.prompt,
                    gen_config,
                    stream=True,
                ):
                    yield f"data: {json.dumps({'text': token})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        else:
            response = await self.engine.generate(request.prompt, gen_config)
            return {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.config.model_path,
                "choices": [{
                    "text": response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    async def _handle_chat_completion(self, request: ChatCompletionRequest) -> Union[Dict, StreamingResponse]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        gen_config = ConfigGenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            num_beams=1,
        )

        if request.stream:
            async def stream_generator():
                async for chunk in self.engine.chat_completion(messages, gen_config, stream=True):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        else:
            return await self.engine.chat_completion(messages, gen_config)

    async def _handle_generate(self, request: GenerateRequest) -> Dict[str, Any]:
        gen_config = ConfigGenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_beams=request.num_beams,
        )

        if request.stream:
            async def stream_generator():
                async for token in self.engine.generate(
                    request.prompt,
                    gen_config,
                    stream=True,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )

        response = await self.engine.generate(request.prompt, gen_config)
        return {"generated_text": response}


# Pydantic models for API
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stream: bool = False


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    stream: bool = False


def create_inference_server(config: InferenceConfig) -> FastAPI:
    """Create FastAPI inference server."""
    server = InferenceServer(config)
    return server.app


def run_server(config: InferenceConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run inference server with uvicorn."""
    import uvicorn
    app = create_inference_server(config)
    uvicorn.run(app, host=host, port=port, workers=1)


__all__ = [
    "InferenceEngine",
    "ChatCompletionEngine",
    "InferenceServer",
    "ModelInstance",
    "InferenceConfig",
    "create_inference_server",
    "run_server",
]