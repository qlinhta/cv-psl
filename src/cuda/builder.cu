#include <torch/torch.h>
#include <torch/script.h>
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <prettytable.h>
#include <tqdm.h>
#include "models.h"
#include "builder.h"
#include "tools.h"
#include "cuda_transforms.h"

namespace fs = std::filesystem;
using namespace std;
using namespace torch;
using namespace torch::data;
using namespace torch::data::datasets;
using namespace torch::nn;
using namespace torch::optim;

class BirdDataset : public datasets::Dataset<BirdDataset> {
public:
    BirdDataset(const std::string& csv_file, const std::string& root_dir, transforms::Normalize& normalize, bool training)
        : root_dir(root_dir), training(training), normalize(normalize) {
        labels_df = cv::ml::TrainData::loadFromCSV(csv_file, 0, -1, -1);
        if (labels_df.empty()) {
            std::cerr << "Failed to load CSV file: " << csv_file << std::endl;
            exit(1);
        }
    }

    Example<> get(size_t index) override {
        auto img_name = fs::path(root_dir) / (labels_df.get(index, 0) + ".jpg");
        cv::Mat image = cv::imread(img_name.string(), cv::IMREAD_COLOR);
        int label = labels_df.get(index, 1);

        torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).permute({2, 0, 1}).to(torch::kCUDA).to(torch::kFloat);
        img_tensor = img_tensor.div(255);

        if (training) {
            img_tensor = cuda_transforms::random_crop(img_tensor, 224, 224);
            if (torch::rand(1).item<float>() > 0.5) {
                img_tensor = cuda_transforms::horizontal_flip(img_tensor);
            }
            img_tensor = cuda_transforms::gaussian_noise(img_tensor, 0.0, 0.1);
        }
        img_tensor = normalize(img_tensor);

        return {img_tensor, torch::tensor(label, torch::kInt64)};
    }

    torch::optional<size_t> size() const override {
        return labels_df.size();
    }

private:
    cv::ml::TrainData labels_df;
    std::string root_dir;
    bool training;
    transforms::Normalize& normalize;
};

std::pair<transforms::Normalize, transforms::Normalize> augment() {
    auto normalize = transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    return {normalize, normalize};
}

std::pair<DataLoader<Example<>>, DataLoader<Example<>>> loader(const std::string& train_csv, const std::string& val_csv, const std::string& train_dir, const std::string& val_dir, int batch_size, int num_workers, transforms::Normalize& train_transform, transforms::Normalize& val_transform) {
    auto train_dataset = BirdDataset(train_csv, train_dir, train_transform, true).map(transforms::Stack<>());
    auto val_dataset = BirdDataset(val_csv, val_dir, val_transform, false).map(transforms::Stack<>());

    auto train_loader = DataLoader<Example<>>(train_dataset, DataLoaderOptions().batch_size(batch_size).workers(num_workers).pin_memory(true).prefetch_factor(4).persistent_workers(true));
    auto val_loader = DataLoader<Example<>>(val_dataset, DataLoaderOptions().batch_size(batch_size).workers(num_workers).pin_memory(true).prefetch_factor(4).persistent_workers(true));

    return {train_loader, val_loader};
}

torch::Device get_device() {
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA" << std::endl;
        torch::cuda::cudnn_is_available();
        return torch::kCUDA;
    } else {
        std::cout << "Using CPU" << std::endl;
        return torch::kCPU;
    }
}
