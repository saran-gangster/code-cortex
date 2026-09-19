#pragma once

#include <string>
#include <vector>

namespace aeroguard {

struct LetterboxTransform {
    int source_width{};
    int source_height{};
    int network_width{};
    int network_height{};
    int resized_width{};
    int resized_height{};
    int pad_x{};
    int pad_y{};
    float scale{};
};

struct Detection {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float confidence{};
    int class_id{};
};

struct Latency {
    float preprocess_ms{};
    float inference_ms{};
    float postprocess_ms{};
    float end_to_end_ms{};
};

struct InferenceOptions {
    float confidence_threshold{0.25F};
    float iou_threshold{0.70F};
    int maximum_detections{300};
};

struct InferenceResult {
    std::vector<Detection> detections;
    Latency latency;
    LetterboxTransform transform;
    std::string input_name;
    std::string output_name;
    std::vector<int> output_shape;
    std::string output_layout;
};

inline constexpr const char* kClassNames[] = {
    "Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"};
inline constexpr int kClassCount = 8;

}  // namespace aeroguard

