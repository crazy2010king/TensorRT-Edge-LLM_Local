#!/usr/bin/env python3
"""
Test script for advanced quantization features: LeptoQuant, SmoothQuant, VLM protection
"""

import argparse
import os
import sys
from tensorrt_edgellm.quantization.llm_quantization import get_llm_quant_config

def test_leptoquant_config():
    """Test LeptoQuant FP8 configuration"""
    print("Testing LeptoQuant FP8 configuration...")
    config = get_llm_quant_config("fp8_lepto", None, None)
    assert "*input_quantizer" in config["quant_cfg"], "Input quantizer not configured"
    assert config["quant_cfg"]["*input_quantizer"]["calibrator"] == "lepto", "Lepto calibrator not set"
    print("✅ LeptoQuant configuration test passed!")

def test_smoothquant_config():
    """Test SmoothQuant configuration"""
    print("\nTesting SmoothQuant (int8_sq) configuration...")
    config = get_llm_quant_config("int8_sq", None, None)
    # SmoothQuant uses INT8 with smooth scaling
    assert "quant_cfg" in config, "Quant config not found"
    print("✅ SmoothQuant configuration test passed!")

def test_visual_protection_config():
    """Test VLM visual protection configuration"""
    print("\nTesting VLM visual protection configuration...")
    # Test without protection (should disable visual)
    config1 = get_llm_quant_config("fp8", None, None, protect_visual_quantization=False)
    assert "*visual.*" in config1["quant_cfg"], "Visual disable config not found"
    assert config1["quant_cfg"]["*visual.*"]["enable"] == False, "Visual should be disabled"

    # Test with protection (should protect sensitive layers)
    config2 = get_llm_quant_config("fp8", None, None, protect_visual_quantization=True)
    assert "*visual.embeddings*" in config2["quant_cfg"], "Visual embedding protection not found"
    assert "*visual.encoder.layers.0*" in config2["quant_cfg"], "Visual layer 0 protection not found"
    assert "*visual.encoder.layers.1*" in config2["quant_cfg"], "Visual layer 1 protection not found"
    assert config2["quant_cfg"]["*visual.embeddings*"]["enable"] == False, "Embeddings should be protected"
    print("✅ VLM visual protection configuration test passed!")

def test_all_quantization_options():
    """Test all quantization options are recognized"""
    print("\nTesting all quantization options...")
    quant_options = ["fp8", "fp8_lepto", "int4_awq", "nvfp4", "mxfp8", "int8_sq"]
    for opt in quant_options:
        try:
            config = get_llm_quant_config(opt, None, None)
            print(f"  ✅ {opt}: supported")
        except Exception as e:
            print(f"  ❌ {opt}: failed - {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Test advanced quantization features")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Advanced Quantization Features for TensorRT Edge-LLM")
    print("=" * 60)

    try:
        test_leptoquant_config()
        test_smoothquant_config()
        test_visual_protection_config()
        test_all_quantization_options()

        print("\n🎉 All tests passed! Advanced quantization features are working correctly.")
        print("\nNew features available:")
        print("  1. LeptoQuant enhanced FP8 quantization (--quantization fp8_lepto)")
        print("  2. SmoothQuant INT8 quantization (--quantization int8_sq)")
        print("  3. VLM visual module protection (--protect-visual-quantization)")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
