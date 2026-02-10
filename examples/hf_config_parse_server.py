import json
import time
from dataclasses import dataclass
from typing import Optional, List

from conduit.blocks import FastAPIServerBlock, TypedRouteSpec, FastAPIServerConfig
from conduit.runtime import LMLiteBlock
from conduit.conduit_types import ComputeProvider, LmLiteModelConfig


@dataclass
class RawHFConfig:
    """
    Client sends the raw contents of Hugging Face config.json as a single string.
    """

    raw_json: str


@dataclass
class LLMVramProfile:
    max_position_embeddings: int

    dtype: str
    quant_dtype: Optional[str]

    num_hidden_layers: int
    num_attention_heads: Optional[int]
    num_kv_heads: int
    head_dim: int
    hidden_size: Optional[int]


# -------------------------
# 2) LLM runtime (the "agent")
# -------------------------

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FP8"  # swap as desired

lm = LMLiteBlock(
    models=[
        LmLiteModelConfig(
            MODEL_ID,
            max_model_len=50_000,
            max_model_concurrency=1,
        )
    ],
    compute_provider=ComputeProvider.RUNPOD,  # or ComputeProvider.RUNPOD
)


GUIDANCE = """

Goal: extract the needed model-geometry + dtype/quant clues using best-effort matching against common patterns. Prefer explicit config values over guesses.

Allowed dtype strings
When you emit/normalize dtype-like values, restrict to this set:
- float32
- float16
- bfloat16
- tf32
- int8
- int4
- fp8
- fp4

1) Where values might live (nesting happens a lot)
Look not only at the top level, but also inside common nested blocks such as:
- model
- text_config
- language_config
- llm_config
- transformer_config
- decoder_config
- architectures-specific subtrees
- quantization_config (often nested)
- bnb_config
- gptq_config
- awq_config
- gguf_config

If both top-level and nested values exist, prefer the one that clearly describes the core transformer (not tokenizer/runtime wrappers).

2) Sequence length / max positions (many aliases)
Configs may express context length under any of:
- max_position_embeddings
- n_positions
- seq_length
- sequence_length
- max_seq_len
- max_sequence_length
- model_max_length

Heuristic: if you see multiple length-like keys, pick the one most directly tied to the model’s positional/attention limit (often max_position_embeddings / n_positions).

3) Layers / heads / hidden size (common naming chaos)
Expect these aliases:

Hidden size:
- hidden_size
- n_embd
- d_model
- dim
- embed_dim
- model_dim

Number of layers:
- num_hidden_layers
- n_layer
- n_layers
- num_layers
- layers
- depth

Attention heads:
- num_attention_heads
- n_head
- attention_heads
- num_heads
- n_heads

KV heads:
- num_key_value_heads
- num_kv_heads
- kv_heads
- n_kv_head

If KV heads are missing, use attention heads (common in non-GQA models).

Be careful: some configs include both encoder_* and decoder_* variants. For decoder-only LLMs, you want the decoder values.

4) Head dimension (explicit or computed)
Head dim may appear as:
- head_dim
- attention_head_size
- head_size

If missing, compute only when safe:
- head_dim = hidden_size / num_attention_heads only if divisible
If not divisible or either value is missing, do not fabricate a number—prefer any explicit head-dim-like key you can find.

5) DType interpretation (BASE dtype vs QUANT dtype)
Treat dtype as two concepts that can coexist:

A) Base dtype (overall / default)
- Prefer torch_dtype if present (this is the model’s default/base precision).
- Other possible base dtype keys: dtype, compute_dtype, weights_dtype, storage_dtype.
- Normalize common synonyms: fp16->float16, bf16->bfloat16, fp32->float32.

B) Quantized dtype (ONLY for quantized modules; never overwrite base torch_dtype)

Output constraint
When emitting/normalizing a quantized dtype, it MUST be exactly one of:
- float32
- float16
- bfloat16
- tf32
- int8
- int4
- fp8
- fp4

Key idea
Quantization configs often describe (method / format / bits) rather than a single dtype.
Your job is to map those signals to ONE concrete quant_dtype from the allowed set.
IMPORTANT: Do NOT overwrite base dtype (torch_dtype) with quant_dtype.

Where to look (quant blocks)
Look inside:
- quantization_config
- bnb_config
- gptq_config
- awq_config
- gguf_config
…and any nested block that contains quant keys (bits, wbits, load_in_4bit, fmt, etc.)

Mapping precedence (highest confidence first)
1) Explicit quant dtype key/value that matches allowed set after normalization
   - Keys: quant_dtype, weight_dtype, w_dtype, dtype (inside quant blocks), storage_dtype
   - Normalize synonyms: fp16->float16, bf16->bfloat16, fp32->float32, float8->fp8
2) Method-based mapping (quant_method / quant_type / checkpoint_format)
3) Bits-based mapping (bits / wbits / w_bit)
4) Backend flags (load_in_8bit / load_in_4bit)
5) Export/container format clues (GGUF quant tags like Q4_*, Q8_*)

If multiple signals exist, choose the most explicit + most directly describing WEIGHT quantization
(not compute/activation dtype).

Method → quant_dtype mapping (deterministic heuristics)

FP8 / Float8
Emit quant_dtype="fp8" if ANY of these indicate FP8:
- quant_method in {"fp8","float8","f8"}
- use_fp8:true
- any of: fp8, fp8_format, float8 present
- fmt in {"e4m3","e5m2","e4m3fn","e5m2fnuz", ...} (any float8 format string)
Example: {"quant_method":"fp8","fmt":"e4m3"} => quant_dtype="fp8"
(Note: fmt is a sub-format; still normalize to fp8 since subtypes aren’t allowed.)

FP4-like
Emit quant_dtype="fp4" if ANY indicate FP4-ish:
- quant_method == "fp4"
- bnb_4bit_quant_type in {"fp4","nf4"}  (treat nf4 as fp4)
- quant_type / weight_quant_type in {"fp4","nf4"}

INT8 / INT4 weight quant
BitsAndBytes:
- load_in_8bit:true => quant_dtype="int8"
- load_in_4bit:true => then:
  - if bnb_4bit_quant_type in {"nf4","fp4"} => quant_dtype="fp4"
  - else => quant_dtype="int4" (fallback)
GPTQ:
- if bits or wbits == 4 => quant_dtype="int4"
- if bits or wbits == 8 => quant_dtype="int8"
- if gptq/gptq_config exists but no bits info => do not guess quant_dtype
AWQ:
- if w_bit == 4 => quant_dtype="int4"
- if w_bit == 8 => quant_dtype="int8"
- if awq/awq_config exists but w_bit missing => do not guess quant_dtype

GGUF / llama.cpp export mapping
Only apply if explicit GGUF indicator exists (gguf:true, gguf_file, file_type, quantization key).
Then map common patterns:
- quant tags starting with "Q4" (Q4_0, Q4_K, Q4_K_M, etc.) => quant_dtype="int4"
- quant tags starting with "Q8" (Q8_0, etc.) => quant_dtype="int8"
- If file_type/quantization explicitly says F16 or BF16 with no other quant indicators:
  treat as NOT quantized (omit quant_dtype); it’s a converted-but-not-quantized export.

Compute dtype is NOT quant dtype
Do NOT treat these as quant_dtype (they describe compute precision, not weight format):
- bnb_4bit_compute_dtype
- compute_dtype
- autocast_dtype
- activation_dtype

Failure rule (don’t fabricate)
If quantization is detected but you cannot map confidently to {int8,int4,fp8,fp4},
prefer omitting quant_dtype rather than inventing one.
Exception: backend flags are strong enough:
- load_in_8bit:true => int8
- load_in_4bit:true => fp4 if nf4/fp4 else int4

""".strip()


# -------------------------
# 3) Route handler (LLM does heavy lifting)
# -------------------------


def build_profile(req: RawHFConfig) -> LLMVramProfile:
    # Optional: very light validation that it's JSON (not doing the heavy parsing here)
    try:
        json.loads(req.raw_json)
    except Exception as e:
        # Raising causes FastAPI to return a 500 by default; if you want a 400,
        # you can adapt FastAPIServerBlock or wrap with your own error protocol.
        raise RuntimeError(f"Invalid JSON: {e}")

    # Ensure deployment is up (optional; can be omitted)
    _ = lm.ready

    profile = lm(
        model_id=MODEL_ID,
        input=req,
        output=LLMVramProfile,
        guidance=GUIDANCE,
    )
    return profile


server = FastAPIServerBlock()

op = server(
    FastAPIServerConfig(
        routes=[
            TypedRouteSpec(
                method="POST",
                path="/v1/profile/from_hf_config",
                input=RawHFConfig,
                output=LLMVramProfile,
                handler=build_profile,
                name="build_llm_vram_profile",
            )
        ],
        host="127.0.0.1",
        port=8080,
        log_level="info",
        ready_timeout_s=10.0,
    )
)

if not op.success:
    raise RuntimeError(op.reason or "Server failed to start")

print(f"Server ready: {op.base_url}")
print(f"POST {op.base_url}/v1/profile/from_hf_config")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    lm.delete()
    server.stop()
