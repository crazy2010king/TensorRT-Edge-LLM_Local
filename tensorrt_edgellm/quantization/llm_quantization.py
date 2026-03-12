# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LLM Quantization Module for TensorRT Edge-LLM.

This module provides quantization utilities for large language models using NVIDIA ModelOpt.
It supports various quantization schemes including FP8, INT4 AWQ, and NVFP4.
"""

import json
import os
import time
from typing import Any, Dict, Optional, Union

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.utils import is_quantized
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq as AutoModelForImageTextToText,
                          AutoTokenizer)

from ..llm_models.model_utils import load_eagle3_draft_model, load_hf_model
from ..llm_models.models.eagle3_draft import Eagle3DraftModel
from .quantization_utils import (enable_huggingface_checkpointing_patch,
                                 quantize_draft_model, quantize_model)

enable_huggingface_checkpointing_patch()

# Quantization configuration constants
# FP8 quantization configuration for language model head.
FP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": None
        },
        "default": {
            "enable": False
        }
    }
}

# INT4 AWQ quantization configuration for language model head.
INT4_AWQ_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.weight_quantizer": {
            "num_bits": 4,
            "block_sizes": {
                -1: 128,
                "type": "static"
            },
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# NVFP4 quantization configuration for language model head.
NVFP4_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {
                -1: 16,
                "type": "dynamic",
                "scale_bits": (4, 3)
            },
            "axis": None,
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# MXFP8 quantization configuration for language model head.
MXFP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "block_sizes": {
                -1: 32,
                "type": "dynamic",
                "scale_bits": (8, 0)
            },
            "enable": True,
        },
        "default": {
            "enable": False
        }
    }
}

# LeptoFP8 quantization configuration for language model head.
LEPTO_FP8_LM_HEAD_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*lm_head.input_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "auto_layer_scale": True
        },
        "*lm_head.weight_quantizer": {
            "num_bits": (4, 3),
            "axis": 0,
            "block_sizes": {-1: 128},
            "auto_layer_scale": True
        },
        "default": {
            "enable": False
        }
    }
}

# Configuration to disable visual model quantization.
DISABLE_VISUAL_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*visual.*": {
            "enable": False
        },
    }
}

# Configuration to protect visual model quantization (only quantize specific layers)
PROTECT_VISUAL_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        # Disable quantization for visual embedding and first 2 layers to preserve accuracy
        "*visual.embeddings*": {"enable": False},
        "*visual.encoder.layers.0*": {"enable": False},
        "*visual.encoder.layers.1*": {"enable": False},
        # Allow quantization for other visual layers
        "*visual.encoder.layers.*": {"enable": True},
    }
}

# LeptoQuant enhanced FP8 quantization configuration
LEPTOQUANT_FP8_CONFIG: Dict[str, Any] = {
    "algorithm": "max",
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (4, 3),
            "axis": -1,
            "calibrator": "max",
            "enable": True
        },
        "*input_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "calibrator": "lepto",
            "enable": True
        },
        "*bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "default": {
            "enable": False
        }
    }
}

# Already merged the vision LoRA and audio is currently not supported, so disable them.
DISABLE_Phi4MM_VISUAL_AUDIO_CONFIG: Dict[str, Any] = {
    "quant_cfg": {
        "*audio_embed.*": {
            "enable": False
        },
        "*image_embed.*": {
            "enable": False
        },
    }
}


def get_llm_calib_dataloader(
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    batch_size: int,
    num_samples: int,
    max_length: int,
) -> DataLoader:
    """
    Create a calibration dataloader for LLM quantization.
    
    Args:
        tokenizer: HuggingFace tokenizer for text processing
        dataset_dir: Dataset name or local directory path
        batch_size: Batch size for the dataloader
        num_samples: Number of samples to use for calibration
        max_length: Maximum sequence length for tokenization
        
    Returns:
        DataLoader: Calibration dataloader with tokenized inputs
        
    Raises:
        NotImplementedError: If dataset format is not supported
    """
    print(f"Loading calibration dataset from {dataset_dir}")
    if "cnn_dailymail" in dataset_dir:
        dataset = load_dataset(dataset_dir, name="3.0.0", split="train")
        dataset = dataset["article"][:num_samples]
    elif os.path.isdir(dataset_dir):
        print(
            f"Recognized local dataset repo {dataset_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_dir, split="train")
        dataset = dataset["text"][:num_samples]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=max_length)

    calib_dataloader = DataLoader(batch_encoded["input_ids"],
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_llm_quant_config(
        quantization: Optional[str], lm_head_quantization: Optional[str],
        kv_cache_quantization: Optional[str],
        smoothquant_alpha: float = 0.5,
        protect_visual_quantization: bool = False) -> Dict[str, Any]:
    """
    Get quantization configuration for LLM models.

    Args:
        quantization: Optional quantization method
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional KV cache quantization method
        smoothquant_alpha: SmoothQuant alpha parameter (0.0-1.0), higher value means more smoothing for activations
        protect_visual_quantization: Whether to protect visual layers from full quantization

    Returns:
        Dict containing quantization configuration

    Raises:
        ValueError: If quantization method is not supported
    """
    # Get base config
    if quantization is None:
        quant_cfg = {"quant_cfg": {}, "algorithm": "max"}
    elif quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "lepto_fp8" or quantization == "fp8_lepto":
        # Support both naming conventions for LeptoQuant FP8
        if quantization == "lepto_fp8":
            # LeptoQuant enhanced FP8 quantization with AutoLayerScale (remote version)
            quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
            # Enable AutoLayerScale for better precision
            quant_cfg["quant_cfg"]["default"]["auto_layer_scale"] = True
            # Optimize weight quantization granularity
            for key in quant_cfg["quant_cfg"]:
                if "weight_quantizer" in key and isinstance(quant_cfg["quant_cfg"][key], dict):
                    quant_cfg["quant_cfg"][key]["axis"] = 0
                    quant_cfg["quant_cfg"][key]["block_sizes"] = {-1: 128}
        else:
            # LeptoQuant enhanced FP8 with lepto calibrator (local version)
            quant_cfg = LEPTOQUANT_FP8_CONFIG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    elif quantization == "w4a8":
        # W4A8 mixed precision quantization: Weight INT4, Activation INT8
        quant_cfg = {
            "quant_cfg": {
                "*weight_quantizer": {
                    "num_bits": 4,
                    "block_sizes": {-1: 128},
                    "enable": True,
                    "type": "weight_only"
                },
                "*input_quantizer": {
                    "num_bits": 8,
                    "enable": True,
                    "axis": None
                },
                "*lm_head.*": {
                    "enable": False
                },
                "default": {
                    "enable": False
                }
            },
            "algorithm": "max"
        }
    elif quantization == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    elif quantization == "mxfp8":
        quant_cfg = mtq.MXFP8_DEFAULT_CFG.copy()
    elif quantization == "int8_sq":
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
        # Set SmoothQuant alpha parameter
        quant_cfg["alpha"] = smoothquant_alpha
    elif quantization == "2bit_tequila":
        # Tequila 2bit ternary quantization
        quant_cfg = {
            "algorithm": "max",
            "quant_cfg": {
                "*weight_quantizer": {
                    "num_bits": 2,
                    "type": "ternary",
                    "axis": -1,
                    "group_size": 128,
                    "enable": True
                },
                "*input_quantizer": {
                    "num_bits": 8,
                    "enable": False  # Keep activations in FP16
                },
                "default": {
                    "enable": False
                }
            }
        }
    elif quantization == "1.25bit_sherry":
        # Sherry 1.25bit quantization
        quant_cfg = {
            "algorithm": "max",
            "quant_cfg": {
                "*weight_quantizer": {
                    "num_bits": 1.25,
                    "type": "symmetric",
                    "axis": -1,
                    "group_size": 256,
                    "enable": True
                },
                "*input_quantizer": {
                    "num_bits": 8,
                    "enable": False  # Keep activations in FP16
                },
                "default": {
                    "enable": False
                }
            }
        }
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    # Add LM head quantization if specified
    if lm_head_quantization is not None:
        # Remove any existing lm_head configuration
        quant_cfg["quant_cfg"] = {
            k: v
            for k, v in quant_cfg["quant_cfg"].items() if "*lm_head" not in k
        }

        if lm_head_quantization == "fp8":
            quant_cfg["quant_cfg"].update(FP8_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "lepto_fp8":
            quant_cfg["quant_cfg"].update(LEPTO_FP8_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "nvfp4":
            quant_cfg["quant_cfg"].update(NVFP4_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "mxfp8":
            quant_cfg["quant_cfg"].update(MXFP8_LM_HEAD_CONFIG["quant_cfg"])

    # Add KV cache quantization if specified
    if kv_cache_quantization is not None:
        if kv_cache_quantization == "fp8":
            quant_cfg["quant_cfg"].update(mtq.FP8_KV_CFG["quant_cfg"])

    # Handle visual model quantization
    if protect_visual_quantization:
        # Protect sensitive visual layers while allowing quantization of others
        quant_cfg["quant_cfg"].update(PROTECT_VISUAL_CONFIG["quant_cfg"])
    else:
        # Disable visual model quantization entirely
        quant_cfg["quant_cfg"].update(DISABLE_VISUAL_CONFIG["quant_cfg"])

    # Disable vision and audio models in Phi-4MM
    quant_cfg["quant_cfg"].update(
        DISABLE_Phi4MM_VISUAL_AUDIO_CONFIG["quant_cfg"])
    return quant_cfg


def quantize_llm(
    model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    quantization: Optional[str],
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
    smoothquant_alpha: float = 0.5,
    protect_visual_quantization: bool = False,
) -> Union[AutoModelForCausalLM, AutoModelForImageTextToText]:
    """
    Quantize a language model using the specified quantization method.

    Args:
        model: The model to quantize (causal LM or image-text model)
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        smoothquant_alpha: SmoothQuant alpha parameter (0.0-1.0), default 0.5
        protect_visual_quantization: Whether to protect visual layers from full quantization
        
    Returns:
        Quantized model
        
    Raises:
        AssertionError: If quantization method is not supported
    """
    assert (quantization is not None) or (lm_head_quantization is not None) or (kv_cache_quantization is not None), \
        "At least one of 'quantization', 'lm_head_quantization', or 'kv_cache_quantization' must be set (not all None)."
    assert quantization in [
        None, "fp8", "lepto_fp8", "fp8_lepto", "int4_awq", "w4a8", "nvfp4", "mxfp8", "int8_sq",
        "2bit_tequila", "1.25bit_sherry"
    ]
    assert lm_head_quantization in [None, "fp8", "lepto_fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # Get calibration dataloader
    if quantization is None or "int4" in quantization:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_llm_calib_dataloader(tokenizer=tokenizer,
                                           dataset_dir=dataset_dir,
                                           batch_size=batch_size,
                                           num_samples=512,
                                           max_length=512)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization, smoothquant_alpha, protect_visual_quantization)
    model = quantize_model(model, quant_config, data_loader)

    return model


def quantize_draft(
    base_model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    draft_model: Union[Eagle3DraftModel],
    tokenizer: AutoTokenizer,
    quantization: str,
    dataset_dir: str,
    lm_head_quantization: Optional[str],
    kv_cache_quantization: Optional[str],
    smoothquant_alpha: float = 0.5,
    protect_visual_quantization: bool = False,
) -> Union[Eagle3DraftModel]:
    """
    Quantize a language model using the specified quantization method.
    
    Args:
        base_model: Based model which is used to generate inputs for the draft model.
        draft_model: The draft model to quantize
        tokenizer: Tokenizer for text processing
        quantization: Quantization method ("fp8", "int4_awq", "nvfp4", "int8_sq")
        dataset_dir: Dataset for calibration
        lm_head_quantization: Optional LM head quantization method
        kv_cache_quantization: Optional KV cache quantization method

    Returns:
        Quantized draft model
        
    Raises:
        AssertionError: If quantization method is not supported
    """
    assert quantization in ["fp8", "lepto_fp8", "fp8_lepto", "int4_awq", "w4a8", "nvfp4", "int8_sq", "mxfp8",
                            "2bit_tequila", "1.25bit_sherry"]
    assert lm_head_quantization in [None, "fp8", "lepto_fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # Get calibration dataloader
    if "int4" in quantization:
        batch_size = 16
    else:
        batch_size = 1
    data_loader = get_llm_calib_dataloader(tokenizer=tokenizer,
                                           dataset_dir=dataset_dir,
                                           batch_size=batch_size,
                                           num_samples=512,
                                           max_length=512)
    quant_config = get_llm_quant_config(quantization, lm_head_quantization,
                                        kv_cache_quantization, smoothquant_alpha, protect_visual_quantization)
    model = quantize_draft_model(base_model, draft_model, quant_config,
                                 data_loader)

    return model


def quantize_and_save_llm(model_dir: str,
                          output_dir: str,
                          quantization: Optional[str] = None,
                          dtype: str = "fp16",
                          dataset_dir: str = "cnn_dailymail",
                          lm_head_quantization: Optional[str] = None,
                          kv_cache_quantization: Optional[str] = None,
                          smoothquant_alpha: float = 0.5,
                          protect_visual_quantization: bool = False,
                          device: str = "cuda") -> None:
    """
    Load a model, quantize it if specified, and save the result.
    
    This is the main entry point for quantizing language models. It supports various
    quantization schemes including FP8, INT4 AWQ, and NVFP4.
    
    Args:
        model_dir: Directory containing the input HuggingFace model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply (None, "fp8", "lepto_fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8")
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for calibration data
        lm_head_quantization: Optional separate quantization for language model head (only "fp8", "nvfp4", and "mxfp8" are currently supported)
        kv_cache_quantization: Optional separate quantization for KV cache (only "fp8" is currently supported)
        smoothquant_alpha: SmoothQuant alpha parameter (0.0-1.0), default 0.5. Higher value reduces activation quantization error but increases weight quantization error.
        device: Device to use for model loading and quantization ("cuda", "cpu")
        
    Raises:
        ValueError: If model loading fails or quantization parameters are invalid
    """
    start_time = time.time()
    # Load model and tokenizer
    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)

    if is_quantized(model):
        print(f"Model is already quantized, skipping quantization.")
    else:
        model = quantize_llm(model, tokenizer, dataset_dir, quantization,
                             lm_head_quantization, kv_cache_quantization, smoothquant_alpha, protect_visual_quantization)

    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if processor is not None:
        processor.save_pretrained(output_dir)

    # Save the quant config
    quant_config = get_quant_config(model)
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(quant_config, f)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")


def quantize_and_save_draft(
    base_model_dir: str,
    draft_model_dir: str,
    output_dir: str,
    quantization: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "fp16",
    dataset_dir: str = "cnn_dailymail",
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    smoothquant_alpha: float = 0.5,
    protect_visual_quantization: bool = False,
) -> None:
    """
    Load an EAGLE draft model, quantize it if specified, and save the result.
    
    This is the main entry point for quantizing EAGLE draft models. It requires
    both a base model and draft model directory.
    
    Args:
        base_model_dir: Directory containing the base HuggingFace model
        draft_model_dir: Directory containing the EAGLE draft model
        output_dir: Directory to save the quantized model
        quantization: Quantization method to apply (None, "fp8", "lepto_fp8", "int4_awq", "nvfp4", "int8_sq", "mxfp8")
        device: Device to use for model loading and quantization ("cuda", "cpu")
        dtype: Model data type for loading ("fp16")
        dataset_dir: Dataset name or path for calibration data
        lm_head_quantization: Optional separate quantization for language model head (only "fp8", "nvfp4", and "mxfp8" are currently supported)
        kv_cache_quantization: Optional separate quantization for KV cache (only "fp8" is currently supported)

    Raises:
        ValueError: If model loading fails or quantization parameters are invalid
    """
    start_time = time.time()

    draft_model = load_eagle3_draft_model(draft_model_dir, base_model_dir,
                                          dtype, device)

    if is_quantized(draft_model):
        print(f"Draft Model is already quantized, skipping quantization.")
    else:
        base_model, tokenizer, _ = load_hf_model(base_model_dir, dtype, device)
        draft_model = quantize_draft(base_model, draft_model, tokenizer,
                                     quantization, dataset_dir,
                                     lm_head_quantization,
                                     kv_cache_quantization,
                                     smoothquant_alpha,
                                     protect_visual_quantization)
    quant_end_time = time.time()
    print(f"Quantization finished in {quant_end_time - start_time}s.")

    # Save the quantized model
    os.makedirs(output_dir, exist_ok=True)

    with torch.inference_mode():
        draft_model.save_pretrained(output_dir)

    # Save the quant config
    quant_config = get_quant_config(draft_model)
    with open(os.path.join(output_dir, "hf_quant_config.json"), "w") as f:
        json.dump(quant_config, f)

    end_time = time.time()
    print(
        f"Quantized model saved to {output_dir} in {end_time - quant_end_time}s."
    )
    print(f"Total time: {end_time - start_time}s.")
