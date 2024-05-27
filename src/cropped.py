import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_yolo_model():
    model = YOLO('yolov8n.pt')
    logging.info("YOLOv8 model loaded successfully.")
    return model


def get_bird_bounding_box_yolo(model, image):
    results = model(image)
    height, width = image.shape[:2]
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'bird':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                if bbox_width > width * 0.3 and bbox_height > height * 0.3:
                    return x1, y1, x2, y2
    return None


def crop_and_resize_image(image, bbox, output_size=(224, 224)):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    max_side = max(width, height)
    new_x1 = max(0, x1 - (max_side - width) // 2)
    new_y1 = max(0, y1 - (max_side - height) // 2)
    new_x2 = min(image.shape[1], new_x1 + max_side)
    new_y2 = min(image.shape[0], new_y1 + max_side)

    if new_x2 - new_x1 < max_side:
        if new_x1 == 0:
            new_x2 = max_side
        else:
            new_x1 = new_x2 - max_side
    if new_y2 - new_y1 < max_side:
        if new_y1 == 0:
            new_y2 = max_side
        else:
            new_y1 = new_y2 - max_side

    cropped_image = image[new_y1:new_y2, new_x1:new_x2]
    resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_AREA)
    return resized_image


def resize_image(image, output_size=(300, 300)):
    resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    return resized_image


def verify_bird_detection(model, image):
    bbox = get_bird_bounding_box_yolo(model, image)
    return bbox is not None


def process_images(input_folder, output_folder, model):
    for subdir in ['train', 'val', 'test']:
        input_subfolder = os.path.join(input_folder, subdir)
        output_subfolder = os.path.join(output_folder, subdir)
        error_file = os.path.join(output_folder, f'{subdir}_errors.csv')

        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        with open(error_file, 'w', newline='') as csvfile:
            error_writer = csv.writer(csvfile)
            error_writer.writerow(['filename'])

            for filename in os.listdir(input_subfolder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(input_subfolder, filename)
                    output_path = os.path.join(output_subfolder, filename)
                    image = cv2.imread(input_path)

                    bbox = get_bird_bounding_box_yolo(model, image)

                    if bbox:
                        processed_image = crop_and_resize_image(image, bbox)
                        if verify_bird_detection(model, processed_image):
                            cv2.imwrite(output_path, processed_image)
                            logging.info(f"Processed and saved image: {output_path}")
                        else:
                            resized_image = resize_image(image)
                            cv2.imwrite(output_path, resized_image)
                            error_writer.writerow([filename])
                            logging.info(f"Post-processing bird detection failed, resized original image: {filename}")
                    else:
                        resized_image = resize_image(image)
                        cv2.imwrite(output_path, resized_image)
                        error_writer.writerow([filename])
                        logging.info(
                            f"Skipped image (no complete bird detected or bird is too small), resized original image: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process bird images to ensure they are centered and resized to 224x224.")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input dataset folder containing train, val, and test subfolders.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder for processed images.')

    args = parser.parse_args()

    model = load_yolo_model()

    logging.info("Starting image processing with YOLOv8 model")
    process_images(args.input, args.output, model)
    logging.info("Image processing completed.")


if __name__ == "__main__":
    main()
