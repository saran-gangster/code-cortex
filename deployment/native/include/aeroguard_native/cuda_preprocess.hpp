#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "aeroguard_native/types.hpp"

namespace aeroguard {

class CudaPreprocessor {
public:
    CudaPreprocessor() = default;
    ~CudaPreprocessor();

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;

    LetterboxTransform run(
        const std::uint8_t* source_bgr,
        int source_width,
        int source_height,
        std::size_t source_stride_bytes,
        void* destination_nchw,
        bool destination_fp16,
        int network_width,
        int network_height,
        cudaStream_t stream);

private:
    void reserve(std::size_t bytes);

    std::uint8_t* pinned_source_{nullptr};
    std::uint8_t* device_source_{nullptr};
    std::size_t capacity_bytes_{0};
};

}  // namespace aeroguard

