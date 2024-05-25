#include <torch/extension.h>
#include <vector>

__global__ void resize_kernel(unsigned char* input, unsigned char* output, int in_width, int in_height, int out_width, int out_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        int in_x = x * in_width / out_width;
        int in_y = y * in_height / out_height;
        int in_idx = (in_y * in_width + in_x) * 3;
        int out_idx = (y * out_width + x) * 3;

        output[out_idx] = input[in_idx];
        output[out_idx + 1] = input[in_idx + 1];
        output[out_idx + 2] = input[in_idx + 2];
    }
}

torch::Tensor resize(torch::Tensor input, int out_width, int out_height) {
    auto input_data = input.data_ptr<unsigned char>();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    auto output = torch::empty({out_height, out_width, 3}, options);
    auto output_data = output.data_ptr<unsigned char>();

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((out_width + 15) / 16, (out_height + 15) / 16);

    resize_kernel<<<num_blocks, threads_per_block>>>(input_data, output_data, input.size(1), input.size(0), out_width, out_height);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resize", &resize, "Resize an image");
}
