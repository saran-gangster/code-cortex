#include "aeroguard_native/cuda_preprocess.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace aeroguard {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ void store_pixel(T* destination, int index, float value);

template <>
__device__ void store_pixel<float>(float* destination, int index, float value) {
    destination[index] = value;
}

template <>
__device__ void store_pixel<__half>(__half* destination, int index, float value) {
    destination[index] = __float2half(value);
}

__device__ float sample_bilinear(
    const std::uint8_t* source,
    int width,
    int height,
    float x,
    float y,
    int channel) {
    x = fminf(fmaxf(x, 0.0F), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0F), static_cast<float>(height - 1));
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const float p00 = static_cast<float>(source[(y0 * width + x0) * 3 + channel]);
    const float p01 = static_cast<float>(source[(y0 * width + x1) * 3 + channel]);
    const float p10 = static_cast<float>(source[(y1 * width + x0) * 3 + channel]);
    const float p11 = static_cast<float>(source[(y1 * width + x1) * 3 + channel]);
    return ((1.0F - wx) * (1.0F - wy) * p00 + wx * (1.0F - wy) * p01 +
            (1.0F - wx) * wy * p10 + wx * wy * p11);
}

template <typename T>
__global__ void letterbox_bgr_to_rgb_nchw(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    T* destination,
    int destination_width,
    int destination_height,
    int resized_width,
    int resized_height,
    int pad_x,
    int pad_y,
    float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) {
        return;
    }

    float red = 114.0F;
    float green = 114.0F;
    float blue = 114.0F;
    if (x >= pad_x && x < pad_x + resized_width && y >= pad_y && y < pad_y + resized_height) {
        const float source_x = (static_cast<float>(x - pad_x) + 0.5F) / scale - 0.5F;
        const float source_y = (static_cast<float>(y - pad_y) + 0.5F) / scale - 0.5F;
        blue = sample_bilinear(source, source_width, source_height, source_x, source_y, 0);
        green = sample_bilinear(source, source_width, source_height, source_x, source_y, 1);
        red = sample_bilinear(source, source_width, source_height, source_x, source_y, 2);
    }

    const int plane = destination_width * destination_height;
    const int offset = y * destination_width + x;
    store_pixel(destination, offset, red / 255.0F);
    store_pixel(destination, plane + offset, green / 255.0F);
    store_pixel(destination, 2 * plane + offset, blue / 255.0F);
}

}  // namespace

CudaPreprocessor::~CudaPreprocessor() {
    if (device_source_ != nullptr) {
        cudaFree(device_source_);
    }
    if (pinned_source_ != nullptr) {
        cudaFreeHost(pinned_source_);
    }
}

void CudaPreprocessor::reserve(std::size_t bytes) {
    if (bytes <= capacity_bytes_) {
        return;
    }
    if (device_source_ != nullptr) {
        check_cuda(cudaFree(device_source_), "cudaFree old image buffer");
        device_source_ = nullptr;
    }
    if (pinned_source_ != nullptr) {
        check_cuda(cudaFreeHost(pinned_source_), "cudaFreeHost old image buffer");
        pinned_source_ = nullptr;
    }
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&pinned_source_), bytes, cudaHostAllocDefault),
               "cudaHostAlloc image buffer");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_source_), bytes), "cudaMalloc image buffer");
    } catch (...) {
        cudaFreeHost(pinned_source_);
        pinned_source_ = nullptr;
        throw;
    }
    capacity_bytes_ = bytes;
}

LetterboxTransform CudaPreprocessor::run(
    const std::uint8_t* source_bgr,
    int source_width,
    int source_height,
    std::size_t source_stride_bytes,
    void* destination_nchw,
    bool destination_fp16,
    int network_width,
    int network_height,
    cudaStream_t stream) {
    if (source_bgr == nullptr || destination_nchw == nullptr || source_width <= 0 || source_height <= 0 ||
        network_width <= 0 || network_height <= 0) {
        throw std::invalid_argument("CUDA preprocessing received invalid image or destination arguments");
    }

    const std::size_t packed_stride = static_cast<std::size_t>(source_width) * 3U;
    const std::size_t bytes = packed_stride * static_cast<std::size_t>(source_height);
    reserve(bytes);
    for (int row = 0; row < source_height; ++row) {
        std::memcpy(
            pinned_source_ + static_cast<std::size_t>(row) * packed_stride,
            source_bgr + static_cast<std::size_t>(row) * source_stride_bytes,
            packed_stride);
    }
    check_cuda(cudaMemcpyAsync(device_source_, pinned_source_, bytes, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync image upload");

    const float scale = std::min(
        static_cast<float>(network_width) / static_cast<float>(source_width),
        static_cast<float>(network_height) / static_cast<float>(source_height));
    const int resized_width = std::max(1, static_cast<int>(std::round(source_width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(source_height * scale)));
    const int pad_x = (network_width - resized_width) / 2;
    const int pad_y = (network_height - resized_height) / 2;

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned>((network_width + block.x - 1) / block.x),
        static_cast<unsigned>((network_height + block.y - 1) / block.y));
    if (destination_fp16) {
        letterbox_bgr_to_rgb_nchw<<<grid, block, 0, stream>>>(
            device_source_, source_width, source_height, static_cast<__half*>(destination_nchw),
            network_width, network_height, resized_width, resized_height, pad_x, pad_y, scale);
    } else {
        letterbox_bgr_to_rgb_nchw<<<grid, block, 0, stream>>>(
            device_source_, source_width, source_height, static_cast<float*>(destination_nchw),
            network_width, network_height, resized_width, resized_height, pad_x, pad_y, scale);
    }
    check_cuda(cudaGetLastError(), "launch letterbox_bgr_to_rgb_nchw");

    return LetterboxTransform{
        source_width,
        source_height,
        network_width,
        network_height,
        resized_width,
        resized_height,
        pad_x,
        pad_y,
        scale};
}

}  // namespace aeroguard

