/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Tequila 2bit ternary quantized GEMV (matrix-vector multiplication)
 *
 * Optimized for batch size 1~4 (M=1~4). Performs: out = in @ W_dequantized
 * where W is 2bit ternary quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K] (Primarily optimized for M ~ [1, 4])
 * @param kernel 2bit quantized weight matrix [N/4, K] in int8 (packed 4x2bit format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void tequila_2bit_gemv_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
    int m, int n, int k, int group_size, cudaStream_t stream);

/*!
 * @brief Tequila 2bit ternary quantized GEMM (matrix-matrix multiplication)
 *
 * Optimized for batch size > 1. Performs: out = in @ W_dequantized
 * where W is 2bit ternary quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K]
 * @param kernel 2bit quantized weight matrix [N/4, K] in int8 (packed 4x2bit format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void tequila_2bit_gemm_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
    int m, int n, int k, int group_size, cudaStream_t stream);

/*!
 * @brief Sherry 1.25bit quantized GEMV (matrix-vector multiplication)
 *
 * Optimized for batch size 1~4 (M=1~4). Performs: out = in @ W_dequantized
 * where W is 1.25bit quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K] (Primarily optimized for M ~ [1, 4])
 * @param kernel 1.25bit quantized weight matrix [N/8, K] in int8 (packed 8x1.25bit format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void sherry_125bit_gemv_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
    int m, int n, int k, int group_size, cudaStream_t stream);

/*!
 * @brief Sherry 1.25bit quantized GEMM (matrix-matrix multiplication)
 *
 * Optimized for batch size > 1. Performs: out = in @ W_dequantized
 * where W is 1.25bit quantized with group-wise scaling factors.
 *
 * @param in_feats Input features [M, K]
 * @param kernel 1.25bit quantized weight matrix [N/8, K] in int8 (packed 8x1.25bit format)
 * @param scaling_factors Group-wise scales [K/group_size, N]
 * @param out_feats Output features [M, N]
 * @param m Batch size
 * @param n Output dimension
 * @param k Input dimension
 * @param group_size Quantization group size
 * @param stream CUDA stream
 */
void sherry_125bit_gemm_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
    int m, int n, int k, int group_size, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
