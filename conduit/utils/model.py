from enum import Enum


class DType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    TF32 = "tf32"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    FP4 = "fp4"


HF_DTYPE_MAP = {
    "float32": DType.FP32,
    "float16": DType.FP16,
    "bfloat16": DType.BF16,
    "tf32": DType.TF32,
    "int8": DType.INT8,
    "int4": DType.INT4,
    "fp8": DType.FP8,
    "fp4": DType.FP4,
}


def bytes_per_dtype(dtype: DType) -> float:
    return {
        DType.FP32: 4.0,
        DType.TF32: 4.0,
        DType.FP16: 2.0,
        DType.BF16: 2.0,
        DType.FP8: 1.0,
        DType.INT8: 1.0,
        DType.FP4: 0.5,
        DType.INT4: 0.5,
    }[dtype]


def map_hf_dtype(dtype: str) -> DType:
    try:
        return HF_DTYPE_MAP[dtype.lower()]
    except KeyError:
        raise ValueError(f"Unsupported HF dtype: {dtype}")
