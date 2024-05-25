#include <torch/extension.h>
#include <vector>
#include <curand_kernel.h>

__global__ void random_crop_kernel(unsigned char* input, unsigned char* output, int in_width, int in_height, int out_width, int out_height, int crop_x, int crop_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        int in_x = x + crop_x;
        int in_y = y + crop_y;
        int in_idx = (in_y * in_width + in_x) * 3;
        int out_idx = (y * out_width + x) * 3;

        output[out_idx] = input[in_idx];
        output[out_idx + 1] = input[in_idx + 1];
        output[out_idx + 2] = input[in_idx + 2];
    }
}

__global__ void horizontal_flip_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx = (y * width + x) * 3;
        int out_idx = (y * width + (width - x - 1)) * 3;

        output[out_idx] = input[in_idx];
        output[out_idx + 1] = input[in_idx + 1];
        output[out_idx + 2] = input[in_idx + 2];
    }
}

__global__ void gaussian_noise_kernel(unsigned char* input, unsigned char* output, int width, int height, float mean, float std, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        int offset = idx * 3;
        curandState local_state = states[idx];
        output[offset] = input[offset] + curand_normal(&local_state) * std + mean;
        output[offset + 1] = input[offset + 1] + curand_normal(&local_state) * std + mean;
        output[offset + 2] = input[offset + 2] + curand_normal(&local_state) * std + mean;
        states[idx] = local_state;
    }
}

__global__ void normalize_kernel(unsigned char* input, float* output, int width, int height, float mean[3], float std[3]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        output[idx] = (input[idx] / 255.0 - mean[0]) / std[0];
        output[idx + 1] = (input[idx + 1] / 255.0 - mean[1]) / std[1];
        output[idx + 2] = (input[idx + 2] / 255.0 - mean[2]) / std[2];
    }
}

torch::Tensor random_crop(torch::Tensor input, int out_width, int out_height) {
    auto input_data = input.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto output = torch::empty({out_height, out_width, 3}, options);
    auto output_data = output.data_ptr<unsigned char>();

    int in_width = input.size(1);
    int in_height = input.size(0);
    int crop_x = rand() % (in_width - out_width + 1);
    int crop_y = rand() % (in_height - out_height + 1);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((out_width + 15) / 16, (out_height + 15) / 16);

    random_crop_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, in_width, in_height, out_width, out_height, crop_x, crop_y);

    return output;
}

torch::Tensor horizontal_flip(torch::Tensor input) {
    auto input_data = input.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto output = torch::empty_like(input);
    auto output_data = output.data_ptr<unsigned char>();

    int width = input.size(1);
    int height = input.size(0);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + 15) / 16, (height + 15) / 16);

    horizontal_flip_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, width, height);

    return output;
}

torch::Tensor gaussian_noise(torch::Tensor input, float mean, float std) {
    auto input_data = input.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto output = torch::empty_like(input);
    auto output_data = output.data_ptr<unsigned char>();

    int width = input.size(1);
    int height = input.size(0);

    curandState* dev_states;
    cudaMalloc((void**)&dev_states, width * height * sizeof(curandState));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + 15) / 16, (height + 15) / 16);

    gaussian_noise_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, width, height, mean, std, dev_states);

    cudaFree(dev_states);

    return output;
}

torch::Tensor normalize(torch::Tensor input, std::vector<float> mean, std::vector<float> std) {
    auto input_data = input.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty_like(input, options);
    auto output_data = output.data_ptr<float>();

    float mean_arr[3] = {mean[0], mean[1], mean[2]};
    float std_arr[3] = {std[0], std[1], std[2]};

    int width = input.size(1);
    int height = input.size(0);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + 15) / 16, (height + 15) / 16);

    normalize_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, width, height, mean_arr, std_arr);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_crop", &random_crop, "Random crop an image");
    m.def("horizontal_flip", &horizontal_flip, "Random horizontal flip an image");
    m.def("gaussian_noise", &gaussian_noise, "Add Gaussian noise to an image");
    m.def("normalize", &normalize, "Normalize an image");
}
