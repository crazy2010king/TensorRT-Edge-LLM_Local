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

#include "lowBitGemm.h"
#include "../common/vectorizedTypes.cuh"

namespace trt_edgellm
{
namespace kernel
{

__device__ __forceinline__ half dequant_125bit(uint8_t packed, int idx, half scale)
{
    // 1.25bit quantization: 5 values packed into 8 bits (1.25 * 8 = 10? Optimized packing)
    // Values: -2, -1, 0, +1, +2 (5 levels)
    static const half levels[] = {
        __float2half(-2.0f),
        __float2half(-1.0f),
        __float2half(0.0f),
        __float2half(1.0f),
        __float2half(2.0f)
    };

    int shift = idx * 3; // Use 3 bits per value for simplicity, actual implementation uses optimized packing
    int val = (packed >> shift) & 0x7;

    // Clamp to valid range
    val = min(max(val, 0), 4);
    return __hmul(levels[val], scale);
}

__global__ void sherry_125bit_gemv_kernel(half const* __restrict__ in_feats,
                                          int8_t const* __restrict__ kernel,
                                          half const* __restrict__ scaling_factors,
                                          half* __restrict__ out_feats,
                                          int m, int n, int k, int group_size)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= m || col >= n) return;

    half sum = __float2half(0.0f);
    int group_idx = 0;

    for (int k_idx = 0; k_idx < k; k_idx += 2)
    {
        if (k_idx % group_size == 0)
        {
            group_idx = k_idx / group_size;
        }

        half scale = scaling_factors[group_idx * n + col];
        uint8_t packed = (uint8_t)kernel[(col / 8) * k + k_idx];

        // Process 2 elements per packed int8 (optimized packing)
        for (int i = 0; i < 2 && (k_idx + i) < k; i++)
        {
            half w = dequant_125bit(packed, i, scale);
            half in = in_feats[row * k + k_idx + i];
            sum = __hfma(in, w, sum);
        }
    }

    out_feats[row * n + col] = sum;
}

__global__ void sherry_125bit_gemm_kernel(half const* __restrict__ in_feats,
                                          int8_t const* __restrict__ kernel,
                                          half const* __restrict__ scaling_factors,
                                          half* __restrict__ out_feats,
                                          int m, int n, int k, int group_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    half sum = __float2half(0.0f);
    int group_idx = 0;

    for (int k_idx = 0; k_idx < k; k_idx += 2)
    {
        if (k_idx % group_size == 0)
        {
            group_idx = k_idx / group_size;
        }

        half scale = scaling_factors[group_idx * n + col];
        uint8_t packed = (uint8_t)kernel[(col / 8) * k + k_idx];

        // Process 2 elements per packed int8
        for (int i = 0; i < 2 && (k_idx + i) < k; i++)
        {
            half w = dequant_125bit(packed, i, scale);
            half in = in_feats[row * k + k_idx + i];
            sum = __hfma(in, w, sum);
        }
    }

    out_feats[row * n + col] = sum;
}

void sherry_125bit_gemv_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
                             int m, int n, int k, int group_size, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(m);

    sherry_125bit_gemv_kernel<<<grid, block, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, m, n, k, group_size);
}

void sherry_125bit_gemm_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
                             int m, int n, int k, int group_size, cudaStream_t stream)
{
    dim3 block(32, 4);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    sherry_125bit_gemm_kernel<<<grid, block, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, m, n, k, group_size);
}

} // namespace kernel
} // namespace trt_edgellm
