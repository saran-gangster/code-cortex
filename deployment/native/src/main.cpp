#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aeroguard_native/tensorrt_runner.hpp"
#include "aeroguard_native/types.hpp"

namespace {

struct Arguments {
    std::string engine;
    std::string image;
    std::string output;
    std::string annotated;
    int image_size{960};
    aeroguard::InferenceOptions inference;
};

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " --engine MODEL.engine --image FRAME.jpg [options]\n"
        << "Options:\n"
        << "  --output FILE.json       Save machine-readable detections\n"
        << "  --annotated FILE.jpg     Save a review image with bounding boxes\n"
        << "  --imgsz PIXELS           Dynamic-profile size (default: 960)\n"
        << "  --conf VALUE             Confidence threshold (default: 0.25)\n"
        << "  --iou VALUE              NMS IoU threshold for raw outputs (default: 0.70)\n"
        << "  --max-det COUNT          Maximum detections (default: 300)\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for argument: " + key);
        }
        const std::string value = argv[++index];
        if (key == "--engine") {
            result.engine = value;
        } else if (key == "--image") {
            result.image = value;
        } else if (key == "--output") {
            result.output = value;
        } else if (key == "--annotated") {
            result.annotated = value;
        } else if (key == "--imgsz") {
            result.image_size = std::stoi(value);
        } else if (key == "--conf") {
            result.inference.confidence_threshold = std::stof(value);
        } else if (key == "--iou") {
            result.inference.iou_threshold = std::stof(value);
        } else if (key == "--max-det") {
            result.inference.maximum_detections = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (result.engine.empty() || result.image.empty()) {
        throw std::invalid_argument("--engine and --image are required");
    }
    return result;
}

std::string escape_json(const std::string& value) {
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default: escaped << character; break;
        }
    }
    return escaped.str();
}

std::string class_name(int class_id) {
    if (class_id < 0 || class_id >= aeroguard::kClassCount) {
        return "Unknown";
    }
    return aeroguard::kClassNames[class_id];
}

std::string to_json(const Arguments& arguments, const aeroguard::InferenceResult& result) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(4);
    output << "{\n"
           << "  \"artifact_kind\": \"aeroguard_native_tensorrt_inference\",\n"
           << "  \"validation_status\": \"experimental_unvalidated\",\n"
           << "  \"engine\": \"" << escape_json(std::filesystem::absolute(arguments.engine).string()) << "\",\n"
           << "  \"image\": \"" << escape_json(std::filesystem::absolute(arguments.image).string()) << "\",\n"
           << "  \"input_name\": \"" << escape_json(result.input_name) << "\",\n"
           << "  \"output_name\": \"" << escape_json(result.output_name) << "\",\n"
           << "  \"output_layout\": \"" << escape_json(result.output_layout) << "\",\n"
           << "  \"output_shape\": [";
    for (std::size_t index = 0; index < result.output_shape.size(); ++index) {
        output << (index == 0 ? "" : ", ") << result.output_shape[index];
    }
    output << "],\n"
           << "  \"latency_ms\": {\n"
           << "    \"preprocess\": " << result.latency.preprocess_ms << ",\n"
           << "    \"inference\": " << result.latency.inference_ms << ",\n"
           << "    \"postprocess\": " << result.latency.postprocess_ms << ",\n"
           << "    \"end_to_end\": " << result.latency.end_to_end_ms << "\n"
           << "  },\n"
           << "  \"detections\": [\n";
    for (std::size_t index = 0; index < result.detections.size(); ++index) {
        const auto& detection = result.detections[index];
        output << "    {\"class_id\": " << detection.class_id
               << ", \"class_name\": \"" << class_name(detection.class_id)
               << "\", \"confidence\": " << detection.confidence
               << ", \"box_xyxy\": [" << detection.x1 << ", " << detection.y1 << ", "
               << detection.x2 << ", " << detection.y2 << "]}"
               << (index + 1 == result.detections.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return output.str();
}

void save_annotated(const cv::Mat& source, const std::vector<aeroguard::Detection>& detections,
                    const std::string& destination) {
    cv::Mat annotated = source.clone();
    for (const auto& detection : detections) {
        const cv::Scalar color = detection.class_id == 0 ? cv::Scalar(120, 235, 255) : cv::Scalar(170, 245, 80);
        cv::rectangle(annotated,
                      cv::Rect(cv::Point(static_cast<int>(detection.x1), static_cast<int>(detection.y1)),
                               cv::Point(static_cast<int>(detection.x2), static_cast<int>(detection.y2))),
                      color, 2, cv::LINE_AA);
        std::ostringstream label;
        label << class_name(detection.class_id) << ' ' << std::fixed << std::setprecision(2)
              << detection.confidence;
        cv::putText(annotated, label.str(),
                    cv::Point(static_cast<int>(detection.x1), std::max(18, static_cast<int>(detection.y1) - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }
    if (!cv::imwrite(destination, annotated)) {
        throw std::runtime_error("OpenCV could not write annotated image: " + destination);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const cv::Mat image = cv::imread(arguments.image, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("OpenCV could not decode image: " + arguments.image);
        }
        aeroguard::TensorRTRunner runner(arguments.engine, arguments.image_size, arguments.image_size);
        const aeroguard::InferenceResult result = runner.infer(image, arguments.inference);
        const std::string json = to_json(arguments, result);
        std::cout << json;
        if (!arguments.output.empty()) {
            std::ofstream output(arguments.output);
            if (!output) {
                throw std::runtime_error("could not open JSON output path: " + arguments.output);
            }
            output << json;
        }
        if (!arguments.annotated.empty()) {
            save_annotated(image, result.detections, arguments.annotated);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AeroGuard native inference failed: " << error.what() << '\n';
        usage(argv[0]);
        return 2;
    }
}

