#include <torch/torch.h>
#include <torch/script.h>
#include <argparse/argparse.hpp>
#include <iostream>
#include <filesystem>
#include <prettytable.h>
#include "models.h"
#include "builder.h"
#include "tools.h"

using namespace std;
using namespace torch;
namespace fs = std::filesystem;

void save_model(torch::nn::Module& model, int epoch, const std::string& model_name, bool best = false) {
    std::string model_dir = "saved_models";
    if (!fs::exists(model_dir)) {
        fs::create_directory(model_dir);
    }
    std::string suffix = best ? "best" : "epoch_" + std::to_string(epoch);
    std::string model_path = model_dir + "/" + model_name + "_" + suffix + ".pt";
    torch::save(model, model_path);
    std::cout << "Model saved to " << model_path << std::endl;
}

void save_batch_images(const torch::Tensor& images, const torch::Tensor& labels, const std::string& phase, int num_images = 8) {
    std::string dump_dir = "dumps/" + phase;
    if (!fs::exists(dump_dir)) {
        fs::create_directory(dump_dir);
    }
    int num_images_to_save = std::min(num_images, images.size(0));
    for (int i = 0; i < num_images_to_save; ++i) {
        torch::save_image(images[i], dump_dir + "/image_" + std::to_string(i) + "_label_" + std::to_string(labels[i].item<int>()) + ".png");
    }
}

void train_model(torch::data::DataLoader<torch::data::Example<>> &train_loader, torch::data::DataLoader<torch::data::Example<>> &val_loader, torch::Device device, int model_id, int num_epochs = 10) {
    auto model_info = get_model_by_id(model_id);
    auto model = model_info.get_model();
    if (torch::cuda::is_available() && torch::cuda::device_count() > 1) {
        std::cout << "Using " << torch::cuda::device_count() << " GPUs" << std::endl;
        model = torch::nn::DataParallel(model);
    }
    model->to(device);
    std::string model_name = model_info.name;

    torch::nn::CrossEntropyLoss criterion;
    torch::optim::AdamW optimizer(model->parameters(), 5.728983638103915e-05);

    std::vector<float> acc_train, acc_val, loss_train, loss_val;
    float best_val_accuracy = 0.0;

    bool train_images_saved = false;
    bool val_images_saved = false;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        float running_loss = 0.0;
        int correct = 0;
        int total = 0;
        for (auto& batch : *train_loader) {
            if (!train_images_saved) {
                save_batch_images(batch.data, batch.target, "train");
                train_images_saved = true;
            }
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);
            optimizer.zero_grad();
            auto outputs = model->forward(images);
            auto loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();
            running_loss += loss.item();
            auto predicted = outputs.argmax(1);
            total += labels.size(0);
            correct += predicted.eq(labels).sum().item();
        }

        float train_loss = running_loss / train_loader->size().value();
        float train_accuracy = 100.0 * correct / total;
        loss_train.push_back(train_loss);
        acc_train.push_back(train_accuracy);

        model->eval();
        float val_loss = 0.0;
        correct = 0;
        total = 0;
        for (auto& batch : *val_loader) {
            if (!val_images_saved) {
                save_batch_images(batch.data, batch.target, "val");
                val_images_saved = true;
            }
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);
            auto outputs = model->forward(images);
            auto loss = criterion(outputs, labels);
            val_loss += loss.item();
            auto predicted = outputs.argmax(1);
            total += labels.size(0);
            correct += predicted.eq(labels).sum().item();
        }

        val_loss = val_loss / val_loader->size().value();
        float val_accuracy = 100.0 * correct / total;
        loss_val.push_back(val_loss);
        acc_val.push_back(val_accuracy);

        prettytable::PrettyTable table;
        table.addRow({"Epoch", "Train Loss", "Val Loss", "Train Accur.", "Val Accur."});
        table.addRow({std::to_string(epoch + 1), std::to_string(train_loss), std::to_string(val_loss), std::to_string(train_accuracy), std::to_string(val_accuracy)});
        std::cout << table;

        if (val_accuracy >= best_val_accuracy) {
            best_val_accuracy = val_accuracy;
            save_model(*model, epoch + 1, model_name, true);
        }
    }

    figure_train_val(model_name, acc_train, acc_val, loss_train, loss_val, true);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser parser("Train bird classification model");
    parser.add_argument("--train_csv").required().help("Path to the train labels CSV file");
    parser.add_argument("--val_csv").required().help("Path to the validation labels CSV file");
    parser.add_argument("--train_dir").required().help("Path to the train images directory");
    parser.add_argument("--val_dir").required().help("Path to the validation images directory");
    parser.add_argument("--batch_size").default_value(32).help("Batch size for the dataloaders");
    parser.add_argument("--num_workers").default_value(8).help("Number of workers for the dataloaders");
    parser.add_argument("--num_epochs").default_value(10).help("Number of epochs for training");
    parser.add_argument("--model_id").required().help("ID of the model to use");
    parser.add_argument("--num_classes").default_value(30).help("Number of output classes");

    auto args = parser.parse_args(argc, argv);

    std::cout << "Using model ID: " << args.get<int>("model_id") << std::endl;
    auto model_info = get_model_by_id(args.get<int>("model_id"));
    std::cout << "Using model: " << model_info.name << std::endl;
    std::cout << "Number of output classes: " << args.get<int>("num_classes") << std::endl;
    std::cout << "Number of epochs: " << args.get<int>("num_epochs") << std::endl;
    std::cout << "Batch size: " << args.get<int>("batch_size") << std::endl;
    std::cout << "Number of workers: " << args.get<int>("num_workers") << std::endl;

    auto [train_transform, val_transform] = augment();

    auto train_loader = loader(args.get<string>("train_csv"), args.get<string>("train_dir"), args.get<int>("batch_size"), args.get<int>("num_workers"), train_transform);
    auto val_loader = loader(args.get<string>("val_csv"), args.get<string>("val_dir"), args.get<int>("batch_size"), args.get<int>("num_workers"), val_transform);

    auto device = device();

    train_model(train_loader, val_loader, device, args.get<int>("model_id"), args.get<int>("num_epochs"));

    return 0;
}
