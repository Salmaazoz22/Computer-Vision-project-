import os
import csv
import cv2
import numpy as np

INPUT_DIR = "processed_dataset"
OUTPUT_DIR = "detections_v2"
CSV_NAME = "detection_results_eval.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_AREA = 300

RED_RANGES = [
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255))
]

BLUE_RANGE = ((90, 60, 40), (140, 255, 255))


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_color_mask(hsv):
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        red_mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

    blue_mask = cv2.inRange(
        hsv,
        np.array(BLUE_RANGE[0], dtype=np.uint8),
        np.array(BLUE_RANGE[1], dtype=np.uint8)
    )

    combined = cv2.bitwise_or(red_mask, blue_mask)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel5)
    return combined


def detect_sign_bbox(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w
    best_box = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        contour = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area
        aspect_ratio = w / float(h)

        if rect_area > image_area * 0.9:
            continue
        if aspect_ratio < 0.6 or aspect_ratio > 1.8:
            continue

        score = rect_area * fill_ratio
        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box, mask


def draw_box(image, box):
    output = image.copy()
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output


def get_all_images(split_dir):
    image_paths = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def process_split(split_name, writer):
    input_split_dir = os.path.join(INPUT_DIR, split_name)
    output_split_dir = os.path.join(OUTPUT_DIR, split_name)
    mask_split_dir = os.path.join(OUTPUT_DIR, f"{split_name}_masks")

    ensure_dir(output_split_dir)
    ensure_dir(mask_split_dir)

    if not os.path.isdir(input_split_dir):
        print(f"Skipped missing folder: {input_split_dir}")
        return

    image_paths = get_all_images(input_split_dir)

    for input_path in image_paths:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read: {input_path}")
            continue

        relative_path = os.path.relpath(input_path, input_split_dir)
        relative_dir = os.path.dirname(relative_path)
        filename = os.path.basename(relative_path)

        save_image_dir = os.path.join(output_split_dir, relative_dir)
        save_mask_dir = os.path.join(mask_split_dir, relative_dir)
        ensure_dir(save_image_dir)
        ensure_dir(save_mask_dir)

        box, mask = detect_sign_bbox(image)
        detected_image = draw_box(image, box)

        output_image_path = os.path.join(save_image_dir, filename)
        output_mask_path = os.path.join(save_mask_dir, filename)

        cv2.imwrite(output_image_path, detected_image)
        cv2.imwrite(output_mask_path, mask)

        csv_image_path = f"{split_name}/{relative_path.replace(os.sep, '/')}"

        if box is not None:
            x, y, w, h = box
            writer.writerow([csv_image_path, x, y, w, h])
            print(f"Detected: {csv_image_path} -> x={x}, y={y}, w={w}, h={h}")
        else:
            image_h, image_w = image.shape[:2]
            writer.writerow([csv_image_path, 0, 0, image_w, image_h])
            print(f"No detection: {csv_image_path}")


def main():
    ensure_dir(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "x", "y", "w", "h"])

        for split_name in ["train", "val", "test"]:
            process_split(split_name, writer)

    print("Done")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"CSV saved in: {csv_path}")


if __name__ == "__main__":
    main()