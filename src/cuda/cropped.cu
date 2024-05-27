#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <cuda_runtime.h>
#include "yolo_v8.h"

using namespace cv;
using namespace std;

#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) { cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; exit(err); } }

void load_yolo_model(YOLOv8 &model) {
    model = YOLOv8("yolov8n.pt");
    cout << "YOLOv8 model loaded successfully." << endl;
}

bool get_bird_bounding_box_yolo(YOLOv8 &model, Mat &image, Rect &bbox) {
    vector<Result> results = model.detect(image);
    int height = image.rows;
    int width = image.cols;
    for (const auto& result : results) {
        if (model.names[result.class_id] == "bird") {
            int x1 = result.bbox.x;
            int y1 = result.bbox.y;
            int x2 = result.bbox.x + result.bbox.width;
            int y2 = result.bbox.y + result.bbox.height;
            int bbox_width = x2 - x1;
            int bbox_height = y2 - y1;
            if (bbox_width > width * 0.3 && bbox_height > height * 0.3) {
                bbox = Rect(x1, y1, bbox_width, bbox_height);
                return true;
            }
        }
    }
    return false;
}

Mat crop_and_resize_image(Mat &image, Rect &bbox, Size output_size = Size(224, 224)) {
    int x1 = bbox.x;
    int y1 = bbox.y;
    int width = bbox.width;
    int height = bbox.height;
    int max_side = max(width, height);
    int new_x1 = max(0, x1 - (max_side - width) / 2);
    int new_y1 = max(0, y1 - (max_side - height) / 2);
    int new_x2 = min(image.cols, new_x1 + max_side);
    int new_y2 = min(image.rows, new_y1 + max_side);

    if (new_x2 - new_x1 < max_side) {
        if (new_x1 == 0) new_x2 = max_side;
        else new_x1 = new_x2 - max_side;
    }
    if (new_y2 - new_y1 < max_side) {
        if (new_y1 == 0) new_y2 = max_side;
        else new_y1 = new_y2 - max_side;
    }

    Mat cropped_image = image(Rect(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1));
    Mat resized_image;
    resize(cropped_image, resized_image, output_size, 0, 0, INTER_AREA);
    return resized_image;
}

Mat resize_image(Mat &image, Size output_size = Size(224, 224)) {
    Mat resized_image;
    resize(image, resized_image, output_size, 0, 0, INTER_AREA);
    return resized_image;
}

bool verify_bird_detection(YOLOv8 &model, Mat &image) {
    Rect bbox;
    return get_bird_bounding_box_yolo(model, image, bbox);
}

void process_images(const string &input_folder, const string &output_folder, YOLOv8 &model) {
    vector<string> subdirs = {"train", "val", "test"};
    for (const string &subdir : subdirs) {
        string input_subfolder = input_folder + "/" + subdir;
        string output_subfolder = output_folder + "/" + subdir;
        string error_file = output_folder + "/" + subdir + "_errors.csv";

        if (mkdir(output_subfolder.c_str(), 0777) == -1 && errno != EEXIST) {
            cerr << "Error creating directory: " << output_subfolder << endl;
            continue;
        }

        ofstream error_csv(error_file);
        error_csv << "filename\n";

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(input_subfolder.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                string filename = ent->d_name;
                if (filename.find(".jpg") != string::npos || filename.find(".jpeg") != string::npos || filename.find(".png") != string::npos) {
                    string input_path = input_subfolder + "/" + filename;
                    string output_path = output_subfolder + "/" + filename;
                    Mat image = imread(input_path);

                    Rect bbox;
                    if (get_bird_bounding_box_yolo(model, image, bbox)) {
                        Mat processed_image = crop_and_resize_image(image, bbox);
                        if (verify_bird_detection(model, processed_image)) {
                            imwrite(output_path, processed_image);
                            cout << "Processed and saved image: " << output_path << endl;
                        } else {
                            Mat resized_image = resize_image(image);
                            imwrite(output_path, resized_image);
                            error_csv << filename << "\n";
                            cout << "Post-processing bird detection failed, resized original image: " << filename << endl;
                        }
                    } else {
                        Mat resized_image = resize_image(image);
                        imwrite(output_path, resized_image);
                        error_csv << filename << "\n";
                        cout << "Skipped image (no complete bird detected or bird is too small), resized original image: " << input_path << endl;
                    }
                }
            }
            closedir(dir);
        } else {
            cerr << "Error opening directory: " << input_subfolder << endl;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_folder> <output_folder>" << endl;
        return 1;
    }

    string input_folder = argv[1];
    string output_folder = argv[2];

    YOLOv8 model;
    load_yolo_model(model);

    cout << "Starting image processing with YOLOv8 model" << endl;
    process_images(input_folder, output_folder, model);
    cout << "Image processing completed." << endl;

    return 0;
}
