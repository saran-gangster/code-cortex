#include "aeroguard_native/tensorrt_runner.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aeroguard_native/cuda_postprocess.hpp"
#include "aeroguard_native/cuda_preprocess.hpp"

namespace aeroguard {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open TensorRT engine: " + path);
    }
    const std::streamsize length = input.tellg();
    if (length <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), length)) {
        throw std::runtime_error("could not read complete TensorRT engine: " + path);
    }
    return bytes;
}

std::pair<const void*, std::size_t> locate_plan(const std::vector<std::uint8_t>& bytes) {
    // Ultralytics prefixes exported .engine files with a little-endian JSON metadata
    // length and the JSON itself. Plain TensorRT plans do not have this prefix.
    if (bytes.size() > 8) {
        std::int32_t metadata_length = 0;
        std::memcpy(&metadata_length, bytes.data(), sizeof(metadata_length));
        const std::size_t length = metadata_length > 0 ? static_cast<std::size_t>(metadata_length) : 0U;
        const std::size_t offset = sizeof(metadata_length) + length;
        if (length > 1 && length < (1U << 20U) && offset < bytes.size() && bytes[4] == '{') {
            return {bytes.data() + offset, bytes.size() - offset};
        }
    }
    return {bytes.data(), bytes.size()};
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(std::uint16_t);
        default:
            throw std::runtime_error("native AeroGuard runner supports only FP32 and FP16 tensors");
    }
}

std::size_t volume(const nvinfer1::Dims& dimensions) {
    if (dimensions.nbDims <= 0) {
        throw std::runtime_error("TensorRT returned a scalar or unresolved tensor shape");
    }
    std::size_t result = 1;
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (dimensions.d[index] <= 0) {
            throw std::runtime_error("TensorRT tensor shape remains dynamic after input shape selection");
        }
        result *= static_cast<std::size_t>(dimensions.d[index]);
    }
    return result;
}

std::vector<int> shape_vector(const nvinfer1::Dims& dimensions) {
    std::vector<int> shape;
    shape.reserve(static_cast<std::size_t>(dimensions.nbDims));
    for (int index = 0; index < dimensions.nbDims; ++index) {
        shape.push_back(dimensions.d[index]);
    }
    return shape;
}

float event_elapsed(cudaEvent_t start, cudaEvent_t end) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, end), "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

class TensorRTRunner::Impl {
public:
    Impl(const std::string& engine_path, int requested_width, int requested_height) {
        if (requested_width <= 0 || requested_height <= 0) {
            throw std::invalid_argument("requested TensorRT dimensions must be positive");
        }
        engine_bytes_ = read_binary(engine_path);
        const auto [plan, plan_bytes] = locate_plan(engine_bytes_);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("TensorRT could not create an inference runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(plan, plan_bytes));
        if (!engine_) {
            throw std::runtime_error(
                "TensorRT could not deserialize this engine. It must match the target JetPack/TensorRT/GPU.");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("TensorRT could not create an execution context");
        }

        for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
            const char* name = engine_->getIOTensorName(index);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) {
                    throw std::runtime_error("native YOLO runner expects exactly one image input tensor");
                }
                input_name_ = name;
            } else {
                if (!output_name_.empty()) {
                    throw std::runtime_error(
                        "native YOLO runner expects one output tensor; this engine uses a multi-output contract");
                }
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("TensorRT engine does not expose one input and one output tensor");
        }

        input_shape_ = engine_->getTensorShape(input_name_.c_str());
        if (input_shape_.nbDims != 4) {
            throw std::runtime_error("image input must have NCHW rank 4");
        }
        input_shape_.d[0] = input_shape_.d[0] < 0 ? 1 : input_shape_.d[0];
        input_shape_.d[1] = input_shape_.d[1] < 0 ? 3 : input_shape_.d[1];
        input_shape_.d[2] = input_shape_.d[2] < 0 ? requested_height : input_shape_.d[2];
        input_shape_.d[3] = input_shape_.d[3] < 0 ? requested_width : input_shape_.d[3];
        if (input_shape_.d[0] != 1 || input_shape_.d[1] != 3) {
            throw std::runtime_error("native edge runner requires a batch-one, three-channel engine");
        }
        if (!context_->setInputShape(input_name_.c_str(), input_shape_)) {
            throw std::runtime_error("requested image dimensions are outside the TensorRT optimization profile");
        }

        output_shape_ = context_->getTensorShape(output_name_.c_str());
        input_type_ = engine_->getTensorDataType(input_name_.c_str());
        output_type_ = engine_->getTensorDataType(output_name_.c_str());
        input_bytes_ = volume(input_shape_) * element_size(input_type_);
        output_bytes_ = volume(output_shape_) * element_size(output_type_);

        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cuda(cudaMalloc(&device_input_, input_bytes_), "cudaMalloc TensorRT input");
        check_cuda(cudaMalloc(&device_output_, output_bytes_), "cudaMalloc TensorRT output");
        if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
            !context_->setTensorAddress(output_name_.c_str(), device_output_)) {
            throw std::runtime_error("TensorRT rejected an input or output device address");
        }
        for (cudaEvent_t* event : {&started_, &preprocessed_, &inferred_, &postprocessed_}) {
            check_cuda(cudaEventCreate(event), "cudaEventCreate");
        }
    }

    ~Impl() {
        for (cudaEvent_t event : {started_, preprocessed_, inferred_, postprocessed_}) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
        if (device_output_ != nullptr) {
            cudaFree(device_output_);
        }
        if (device_input_ != nullptr) {
            cudaFree(device_input_);
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    InferenceResult infer(const cv::Mat& image, const InferenceOptions& options) {
        if (image.empty() || image.type() != CV_8UC3) {
            throw std::invalid_argument("inference image must be a non-empty OpenCV CV_8UC3 BGR matrix");
        }
        const auto wall_start = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(started_, stream_), "record inference start");
        const LetterboxTransform transform = preprocessor_.run(
            image.ptr<std::uint8_t>(), image.cols, image.rows, image.step, device_input_,
            input_type_ == nvinfer1::DataType::kHALF, input_shape_.d[3], input_shape_.d[2], stream_);
        check_cuda(cudaEventRecord(preprocessed_, stream_), "record preprocessing completion");

        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 returned failure");
        }
        check_cuda(cudaEventRecord(inferred_, stream_), "record TensorRT completion");

        PostprocessResult postprocessed = postprocess_yolo_cuda(
            OutputTensorView{
                device_output_, output_type_ == nvinfer1::DataType::kHALF, shape_vector(output_shape_)},
            transform,
            options,
            stream_);
        check_cuda(cudaEventRecord(postprocessed_, stream_), "record postprocessing completion");
        check_cuda(cudaEventSynchronize(postprocessed_), "synchronize native inference");
        const auto wall_end = std::chrono::steady_clock::now();

        InferenceResult result;
        result.detections = std::move(postprocessed.detections);
        result.latency.preprocess_ms = event_elapsed(started_, preprocessed_);
        result.latency.inference_ms = event_elapsed(preprocessed_, inferred_);
        result.latency.postprocess_ms = event_elapsed(inferred_, postprocessed_);
        result.latency.end_to_end_ms =
            std::chrono::duration<float, std::milli>(wall_end - wall_start).count();
        result.transform = transform;
        result.input_name = input_name_;
        result.output_name = output_name_;
        result.output_shape = shape_vector(output_shape_);
        result.output_layout = std::move(postprocessed.layout);
        return result;
    }

private:
    Logger logger_;
    std::vector<std::uint8_t> engine_bytes_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string input_name_;
    std::string output_name_;
    nvinfer1::Dims input_shape_{};
    nvinfer1::Dims output_shape_{};
    nvinfer1::DataType input_type_{};
    nvinfer1::DataType output_type_{};
    std::size_t input_bytes_{};
    std::size_t output_bytes_{};
    void* device_input_{nullptr};
    void* device_output_{nullptr};
    cudaStream_t stream_{nullptr};
    cudaEvent_t started_{nullptr};
    cudaEvent_t preprocessed_{nullptr};
    cudaEvent_t inferred_{nullptr};
    cudaEvent_t postprocessed_{nullptr};
    CudaPreprocessor preprocessor_;
};

TensorRTRunner::TensorRTRunner(const std::string& engine_path, int requested_width, int requested_height)
    : impl_(std::make_unique<Impl>(engine_path, requested_width, requested_height)) {}

TensorRTRunner::~TensorRTRunner() = default;

InferenceResult TensorRTRunner::infer(const cv::Mat& bgr_image, const InferenceOptions& options) {
    return impl_->infer(bgr_image, options);
}

}  // namespace aeroguard

