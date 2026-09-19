#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "aeroguard_native/types.hpp"

namespace aeroguard {

struct OutputTensorView {
    const void* device_data{};
    bool fp16{};
    std::vector<int> shape;
};

struct PostprocessResult {
    std::vector<Detection> detections;
    std::string layout;
};

PostprocessResult postprocess_yolo_cuda(
    const OutputTensorView& output,
    const LetterboxTransform& transform,
    const InferenceOptions& options,
    cudaStream_t stream);

}  // namespace aeroguard

