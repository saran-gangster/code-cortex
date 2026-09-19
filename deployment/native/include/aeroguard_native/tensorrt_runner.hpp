#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "aeroguard_native/types.hpp"

namespace aeroguard {

class TensorRTRunner {
public:
    TensorRTRunner(const std::string& engine_path, int requested_width = 960, int requested_height = 960);
    ~TensorRTRunner();

    TensorRTRunner(const TensorRTRunner&) = delete;
    TensorRTRunner& operator=(const TensorRTRunner&) = delete;

    InferenceResult infer(const cv::Mat& bgr_image, const InferenceOptions& options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aeroguard

