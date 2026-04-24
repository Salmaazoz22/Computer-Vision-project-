import os
import csv
import cv2

INPUT_DIR = "processed_dataset"
OUTPUT_DIR = "detections_v2"
CSV_NAME = "pseudo_ground_truth.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_all_images(split_dir):
    image_paths = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def create_pseudo_box(image_w, image_h):
    x = int(image_w * 0.15)
    y = int(image_h * 0.15)
    w = int(image_w * 0.70)
    h = int(image_h * 0.70)
    return x, y, w, h


def process_split(split_name, writer):
    input_split_dir = os.path.join(INPUT_DIR, split_name)

    if not os.path.isdir(input_split_dir):
        print(f"Skipped missing folder: {input_split_dir}")
        return

    image_paths = get_all_images(input_split_dir)

    for input_path in image_paths:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read: {input_path}")
            continue

        image_h, image_w = image.shape[:2]
        x, y, w, h = create_pseudo_box(image_w, image_h)

        relative_path = os.path.relpath(input_path, input_split_dir)
        csv_image_path = f"{split_name}/{relative_path.replace(os.sep, '/')}"

        writer.writerow([csv_image_path, x, y, w, h])
        print(f"Pseudo GT: {csv_image_path} -> x={x}, y={y}, w={w}, h={h}")


def main():
    ensure_dir(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "x", "y", "w", "h"])

        for split_name in ["train", "val", "test"]:
            process_split(split_name, writer)

    print("Done")
    print(f"Pseudo Ground Truth saved in: {csv_path}")


if __name__ == "__main__":
    main()