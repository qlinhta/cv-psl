import os
import argparse
from PIL import Image
from ultralytics import YOLO
from builder import device
import logging
import coloredlogs

logger = logging.getLogger(__name__)
log_file_handler = logging.FileHandler('preprocessing.log')
log_file_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_file_handler.setFormatter(log_formatter)
logger.addHandler(log_file_handler)

coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def __get_device__():
    return device()


def load_model():
    logger.info("Loading YOLOv8 model")
    return YOLO('yolov8x.pt').to(__get_device__())


def expand_bounding_box(box, width, height, margin_ratio=0.1):
    x_min, y_min, x_max, y_max = box

    margin_x = (x_max - x_min) * margin_ratio
    margin_y = (y_max - y_min) * margin_ratio

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(width, x_max + margin_x)
    y_max = min(height, y_max + margin_y)

    return x_min, y_min, x_max, y_max


def detect_and_crop(model, image_path, output_path, margin_ratio=0.1):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        box = boxes[0]
        x_min, y_min, x_max, y_max = expand_bounding_box(box, width, height, margin_ratio)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image.thumbnail((224, 224), Image.LANCZOS)
        cropped_image.save(output_path)


def preprocess_folder(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    logger.info(f"Processing {len(image_files)} images in {input_folder}")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        try:
            detect_and_crop(model, image_path, output_path)
            logger.info(f"Processed {image_file}")
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")


def preprocess_dataset(dataset_folder, output_folder):
    model = load_model()

    for phase in ['train', 'val']:
        input_folder = os.path.join(dataset_folder, phase)
        output_phase_folder = os.path.join(output_folder, phase)
        logger.info(f"Processing {phase} set")
        preprocess_folder(input_folder, output_phase_folder, model)
        logger.info(f"Finished processing {phase} set")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset by detecting and cropping birds.")
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help="Path to the dataset containing 'train', 'val', 'test' folders")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to save the cropped dataset")

    args = parser.parse_args()

    preprocess_dataset(args.dataset_folder, args.output_folder)
