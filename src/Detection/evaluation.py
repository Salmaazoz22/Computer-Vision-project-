import pandas as pd

pred_path = "detections_v2/detection_results_eval.csv"
gt_path = "detections_v2/pseudo_ground_truth.csv"

pred_df = pd.read_csv(pred_path)
gt_df = pd.read_csv(gt_path)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    union = areaA + areaB - inter_area

    if union == 0:
        return 0

    return inter_area / union

ious = []

for i in range(len(pred_df)):
    pred_row = pred_df.iloc[i]
    gt_row = gt_df.iloc[i]

    pred_box = (pred_row["x"], pred_row["y"], pred_row["w"], pred_row["h"])
    gt_box = (gt_row["x"], gt_row["y"], gt_row["w"], gt_row["h"])

    iou = calculate_iou(pred_box, gt_box)
    ious.append(iou)

average_iou = sum(ious) / len(ious)

print("Average IoU:", average_iou)