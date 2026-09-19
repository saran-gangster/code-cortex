#include "aeroguard_native/cuda_postprocess.hpp"

#include <cuda_fp16.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeroguard {
namespace {

constexpr int kThreads = 256;
constexpr int kNmsBlock = 64;
constexpr int kMaximumNmsCandidates = 4096;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ float read_value(const T* values, int index) {
    return static_cast<float>(values[index]);
}

template <>
__device__ float read_value<__half>(const __half* values, int index) {
    return __half2float(values[index]);
}

__device__ Detection restore_box(
    float x1,
    float y1,
    float x2,
    float y2,
    float confidence,
    int class_id,
    LetterboxTransform transform) {
    Detection result;
    result.x1 = fminf(fmaxf((x1 - transform.pad_x) / transform.scale, 0.0F), transform.source_width - 1.0F);
    result.y1 = fminf(fmaxf((y1 - transform.pad_y) / transform.scale, 0.0F), transform.source_height - 1.0F);
    result.x2 = fminf(fmaxf((x2 - transform.pad_x) / transform.scale, 0.0F), transform.source_width - 1.0F);
    result.y2 = fminf(fmaxf((y2 - transform.pad_y) / transform.scale, 0.0F), transform.source_height - 1.0F);
    result.confidence = confidence;
    result.class_id = class_id;
    return result;
}

template <typename T>
__global__ void filter_end_to_end(
    const T* output,
    int rows,
    float threshold,
    LetterboxTransform transform,
    Detection* candidates,
    int* count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows) {
        return;
    }
    const int offset = index * 6;
    const float confidence = read_value(output, offset + 4);
    const int class_id = static_cast<int>(read_value(output, offset + 5));
    if (confidence < threshold || class_id < 0 || class_id >= kClassCount) {
        return;
    }
    const int destination = atomicAdd(count, 1);
    candidates[destination] = restore_box(
        read_value(output, offset), read_value(output, offset + 1),
        read_value(output, offset + 2), read_value(output, offset + 3),
        confidence, class_id, transform);
}

template <typename T>
__global__ void decode_raw_channels_first(
    const T* output,
    int channels,
    int predictions,
    float threshold,
    LetterboxTransform transform,
    Detection* candidates,
    int* count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= predictions) {
        return;
    }
    int best_class = 0;
    float best_score = read_value(output, 4 * predictions + index);
    for (int class_id = 1; class_id < channels - 4; ++class_id) {
        const float score = read_value(output, (4 + class_id) * predictions + index);
        if (score > best_score) {
            best_score = score;
            best_class = class_id;
        }
    }
    if (best_score < threshold || best_class >= kClassCount) {
        return;
    }
    const float cx = read_value(output, index);
    const float cy = read_value(output, predictions + index);
    const float width = read_value(output, 2 * predictions + index);
    const float height = read_value(output, 3 * predictions + index);
    const int destination = atomicAdd(count, 1);
    candidates[destination] = restore_box(
        cx - width * 0.5F, cy - height * 0.5F, cx + width * 0.5F, cy + height * 0.5F,
        best_score, best_class, transform);
}

template <typename T>
__global__ void decode_raw_channels_last(
    const T* output,
    int channels,
    int predictions,
    float threshold,
    LetterboxTransform transform,
    Detection* candidates,
    int* count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= predictions) {
        return;
    }
    const int offset = index * channels;
    int best_class = 0;
    float best_score = read_value(output, offset + 4);
    for (int class_id = 1; class_id < channels - 4; ++class_id) {
        const float score = read_value(output, offset + 4 + class_id);
        if (score > best_score) {
            best_score = score;
            best_class = class_id;
        }
    }
    if (best_score < threshold || best_class >= kClassCount) {
        return;
    }
    const float cx = read_value(output, offset);
    const float cy = read_value(output, offset + 1);
    const float width = read_value(output, offset + 2);
    const float height = read_value(output, offset + 3);
    const int destination = atomicAdd(count, 1);
    candidates[destination] = restore_box(
        cx - width * 0.5F, cy - height * 0.5F, cx + width * 0.5F, cy + height * 0.5F,
        best_score, best_class, transform);
}

struct ConfidenceDescending {
    __host__ __device__ bool operator()(const Detection& left, const Detection& right) const {
        return left.confidence > right.confidence;
    }
};

__device__ float intersection_over_union(const Detection& left, const Detection& right) {
    const float x1 = fmaxf(left.x1, right.x1);
    const float y1 = fmaxf(left.y1, right.y1);
    const float x2 = fminf(left.x2, right.x2);
    const float y2 = fminf(left.y2, right.y2);
    const float intersection = fmaxf(0.0F, x2 - x1) * fmaxf(0.0F, y2 - y1);
    const float left_area = fmaxf(0.0F, left.x2 - left.x1) * fmaxf(0.0F, left.y2 - left.y1);
    const float right_area = fmaxf(0.0F, right.x2 - right.x1) * fmaxf(0.0F, right.y2 - right.y1);
    return intersection / fmaxf(left_area + right_area - intersection, 1.0e-7F);
}

__global__ void build_nms_mask(
    const Detection* candidates,
    int candidate_count,
    float iou_threshold,
    std::uint64_t* mask,
    int column_blocks) {
    const int row = blockIdx.y * kNmsBlock + threadIdx.x;
    const int column_start = blockIdx.x * kNmsBlock;
    __shared__ Detection column_boxes[kNmsBlock];
    const int column_index = column_start + threadIdx.x;
    if (column_index < candidate_count) {
        column_boxes[threadIdx.x] = candidates[column_index];
    }
    __syncthreads();
    if (row >= candidate_count) {
        return;
    }

    std::uint64_t bits = 0;
    const int limit = min(kNmsBlock, candidate_count - column_start);
    for (int offset = 0; offset < limit; ++offset) {
        const int other = column_start + offset;
        if (other > row && candidates[row].class_id == column_boxes[offset].class_id &&
            intersection_over_union(candidates[row], column_boxes[offset]) > iou_threshold) {
            bits |= std::uint64_t{1} << offset;
        }
    }
    mask[static_cast<std::size_t>(row) * column_blocks + blockIdx.x] = bits;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "cudaMalloc postprocess buffer");
        }
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_{nullptr};
    std::size_t count_{};
};

template <typename T>
void launch_decode(
    const OutputTensorView& output,
    bool end_to_end,
    bool channels_first,
    int channels,
    int predictions,
    const LetterboxTransform& transform,
    float threshold,
    Detection* candidates,
    int* count,
    cudaStream_t stream) {
    const int blocks = (predictions + kThreads - 1) / kThreads;
    const T* values = static_cast<const T*>(output.device_data);
    if (end_to_end) {
        filter_end_to_end<<<blocks, kThreads, 0, stream>>>(
            values, predictions, threshold, transform, candidates, count);
    } else if (channels_first) {
        decode_raw_channels_first<<<blocks, kThreads, 0, stream>>>(
            values, channels, predictions, threshold, transform, candidates, count);
    } else {
        decode_raw_channels_last<<<blocks, kThreads, 0, stream>>>(
            values, channels, predictions, threshold, transform, candidates, count);
    }
    check_cuda(cudaGetLastError(), "launch YOLO decode kernel");
}

}  // namespace

PostprocessResult postprocess_yolo_cuda(
    const OutputTensorView& output,
    const LetterboxTransform& transform,
    const InferenceOptions& options,
    cudaStream_t stream) {
    if (output.device_data == nullptr || output.shape.empty()) {
        throw std::invalid_argument("YOLO postprocessing received an empty output tensor");
    }
    if (options.confidence_threshold < 0.0F || options.confidence_threshold > 1.0F ||
        options.iou_threshold < 0.0F || options.iou_threshold > 1.0F || options.maximum_detections <= 0) {
        throw std::invalid_argument("invalid confidence, IoU, or maximum-detections option");
    }

    const int rank = static_cast<int>(output.shape.size());
    const bool end_to_end = rank >= 2 && output.shape.back() == 6;
    bool channels_first = false;
    int channels = 6;
    int predictions = 0;
    if (end_to_end) {
        predictions = output.shape[rank - 2];
    } else if (rank == 3 && output.shape[1] == 4 + kClassCount) {
        channels_first = true;
        channels = output.shape[1];
        predictions = output.shape[2];
    } else if (rank == 3 && output.shape[2] == 4 + kClassCount) {
        channels = output.shape[2];
        predictions = output.shape[1];
    } else {
        throw std::runtime_error(
            "unsupported YOLO output shape; expected [1,N,6], [1,12,N], or [1,N,12]");
    }
    if (predictions <= 0) {
        throw std::runtime_error("YOLO output contains no prediction rows");
    }

    DeviceBuffer<Detection> device_candidates(static_cast<std::size_t>(predictions));
    DeviceBuffer<int> device_count(1);
    check_cuda(cudaMemsetAsync(device_count.get(), 0, sizeof(int), stream), "clear candidate count");
    if (output.fp16) {
        launch_decode<__half>(output, end_to_end, channels_first, channels, predictions, transform,
                              options.confidence_threshold, device_candidates.get(), device_count.get(), stream);
    } else {
        launch_decode<float>(output, end_to_end, channels_first, channels, predictions, transform,
                             options.confidence_threshold, device_candidates.get(), device_count.get(), stream);
    }

    int candidate_count = 0;
    check_cuda(cudaMemcpyAsync(&candidate_count, device_count.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
               "copy candidate count");
    check_cuda(cudaStreamSynchronize(stream), "synchronize decoded candidates");
    candidate_count = std::min(candidate_count, predictions);
    if (candidate_count == 0) {
        return {{}, end_to_end ? "end_to_end_xyxy" : "raw_xywh_cuda_nms"};
    }

    thrust::device_ptr<Detection> begin(device_candidates.get());
    thrust::sort(thrust::cuda::par.on(stream), begin, begin + candidate_count, ConfidenceDescending{});

    if (end_to_end) {
        const int returned = std::min(candidate_count, options.maximum_detections);
        std::vector<Detection> host(static_cast<std::size_t>(returned));
        check_cuda(cudaMemcpyAsync(host.data(), device_candidates.get(), host.size() * sizeof(Detection),
                                   cudaMemcpyDeviceToHost, stream), "copy end-to-end detections");
        check_cuda(cudaStreamSynchronize(stream), "synchronize end-to-end detections");
        return {std::move(host), "end_to_end_xyxy"};
    }

    const int nms_count = std::min(candidate_count, kMaximumNmsCandidates);
    const int column_blocks = (nms_count + kNmsBlock - 1) / kNmsBlock;
    DeviceBuffer<std::uint64_t> device_mask(static_cast<std::size_t>(nms_count) * column_blocks);
    const dim3 grid(static_cast<unsigned>(column_blocks), static_cast<unsigned>(column_blocks));
    build_nms_mask<<<grid, kNmsBlock, 0, stream>>>(
        device_candidates.get(), nms_count, options.iou_threshold, device_mask.get(), column_blocks);
    check_cuda(cudaGetLastError(), "launch class-aware CUDA NMS mask kernel");

    std::vector<Detection> sorted(static_cast<std::size_t>(nms_count));
    std::vector<std::uint64_t> mask(static_cast<std::size_t>(nms_count) * column_blocks);
    check_cuda(cudaMemcpyAsync(sorted.data(), device_candidates.get(), sorted.size() * sizeof(Detection),
                               cudaMemcpyDeviceToHost, stream), "copy sorted candidates");
    check_cuda(cudaMemcpyAsync(mask.data(), device_mask.get(), mask.size() * sizeof(std::uint64_t),
                               cudaMemcpyDeviceToHost, stream), "copy NMS mask");
    check_cuda(cudaStreamSynchronize(stream), "synchronize CUDA NMS");

    std::vector<std::uint64_t> removed(static_cast<std::size_t>(column_blocks), 0);
    std::vector<Detection> selected;
    selected.reserve(static_cast<std::size_t>(std::min(nms_count, options.maximum_detections)));
    for (int index = 0; index < nms_count && static_cast<int>(selected.size()) < options.maximum_detections; ++index) {
        const int block = index / kNmsBlock;
        const int bit = index % kNmsBlock;
        if ((removed[block] & (std::uint64_t{1} << bit)) != 0) {
            continue;
        }
        selected.push_back(sorted[index]);
        const std::uint64_t* row = mask.data() + static_cast<std::size_t>(index) * column_blocks;
        for (int column = block; column < column_blocks; ++column) {
            removed[column] |= row[column];
        }
    }
    return {std::move(selected), "raw_xywh_cuda_nms"};
}

}  // namespace aeroguard

