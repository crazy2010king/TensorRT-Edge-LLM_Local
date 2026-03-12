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

__device__ __forceinline__ half dequant_2bit(int8_t packed, int idx, half scale)
{
    // Extract 2-bit value: 00=+1, 01=-1, 10=+0.5, 11=-0.5 (ternary values)
    int val = (packed >> (idx * 2)) & 0x3;
    half res;
    switch (val)
    {
        case 0: res = __float2half(1.0f); break;
        case 1: res = __float2half(-1.0f); break;
        case 2: res = __float2half(0.5f); break;
        case 3: res = __float2half(-0.5f); break;
        default: res = __float2half(0.0f);
    }
    return __hmul(res, scale);
}

__global__ void tequila_2bit_gemv_kernel(half const* __restrict__ in_feats,
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

    for (int k_idx = 0; k_idx < k; k_idx += 4)
    {
        if (k_idx % group_size == 0)
        {
            group_idx = k_idx / group_size;
        }

        half scale = scaling_factors[group_idx * n + col];
        int8_t packed = kernel[(col / 4) * k + k_idx];

        // Process 4 elements per packed int8
        for (int i = 0; i < 4 && (k_idx + i) < k; i++)
        {
            half w = dequant_2bit(packed, i, scale);
            half in = in_feats[row * k + k_idx + i];
            sum = __hfma(in, w, sum);
        }
    }

    out_feats[row * n + col] = sum;
}

__global__ void tequila_2bit_gemm_kernel(half const* __restrict__ in_feats,
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

    for (int k_idx = 0; k_idx < k; k_idx += 4)
    {
        if (k_idx % group_size == 0)
        {
            group_idx = k_idx / group_size;
        }

        half scale = scaling_factors[group_idx * n + col];
        int8_t packed = kernel[(col / 4) * k + k_idx];

        // Process 4 elements per packed int8
        for (int i = 0; i < 4 && (k_idx + i) < k; i++)
        {
            half w = dequant_2bit(packed, i, scale);
            half in = in_feats[row * k + k_idx + i];
            sum = __hfma(in, w, sum);
        }
    }

    out_feats[row * n + col] = sum;
}

void tequila_2bit_gemv_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
                            int m, int n, int k, int group_size, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid(m);

    tequila_2bit_gemv_kernel<<<grid, block, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, m, n, k, group_size);
}

void tequila_2bit_gemm_cuda(half const* in_feats, int8_t const* kernel, half const* scaling_factors, half* out_feats,
                            int m, int n, int k, int group_size, cudaStream_t stream)
{
    dim3 block(32, 4);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    tequila_2bit_gemm_kernel<<<grid, block, 0, stream>>>(in_feats, kernel, scaling_factors, out_feats, m, n, k, group_size);
}

} // namespace kernel
} // namespace trt_edgellm
