import os
import numpy as np
from sklearn.metrics import average_precision_score
import torch

# 假设你的路径如下
# txt_folder = r'C:\Users\a\Desktop\tmptrack'  # 替换为你的txt文件路径
txt_folder = r'E:\cv\RTMOT\output\train\eval_during_train\val\epoch_2\tracker'  # 替换为你的txt文件路径
gt_base_folder = r'E:\data\DanceTrack\val'  # 替换为同名文件夹的路径


def read_mot_txt(file_path):
    # 读取 MOT 格式的 txt 文件
    data = np.loadtxt(file_path, delimiter=',')
    return data


def read_gt(gt_path):
    # 读取 GT 文件
    gt_data = np.loadtxt(gt_path, delimiter=',')
    return gt_data


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Computes the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Computes the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def xywh2xyxy(x):
    # 将 (x_center, y_center, width, height) 转换为 (x1, y1, x2, y2)
    x1 = x[:, 0] - x[:, 2] / 2
    y1 = x[:, 1] - x[:, 3] / 2
    x2 = x[:, 0] + x[:, 2]
    y2 = x[:, 1] + x[:, 3]
    return np.stack((x1, y1, x2, y2), axis=1)


def calculate_iou(box1, box2):
    # 计算交并比 (IoU)
    x1_inter = np.maximum(box1[0], box2[:, 0])
    y1_inter = np.maximum(box1[1], box2[:, 1])
    x2_inter = np.minimum(box1[2], box2[:, 2])
    y2_inter = np.minimum(box1[3], box2[:, 3])

    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def main():
    mAPs = []
    mR = []
    mP = []

    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt') and 'dance' in txt_file:
            preds = read_mot_txt(os.path.join(txt_folder, txt_file))

            # 构建 GT 文件路径
            folder_name = os.path.splitext(txt_file)[0]  # 获取文件名不带扩展名
            gt_folder_path = os.path.join(gt_base_folder, folder_name, 'gt')
            gt_file_path = os.path.join(gt_folder_path, 'gt.txt')

            gts = read_gt(gt_file_path)

            for frame in np.unique(preds[:, 0]):
                frame_preds = preds[preds[:, 0] == frame]
                frame_gts = gts[gts[:, 0] == frame]

                if frame_preds.size == 0:
                    mAPs.append(0)
                    mR.append(0)
                    mP.append(0)
                    continue

                # 获取置信度并排序
                frame_preds = frame_preds[np.argsort(-frame_preds[:, 4])]  # 假设第5列是置信度
                correct = []
                confs = []
                clses = []
                detected = []
                cls_gts = [1 for _ in range(len(frame_preds))]
                if frame_gts.size == 0:
                    correct = [0] * len(frame_preds)
                    confs = [0] * len(frame_preds)
                    clses = [0] * len(frame_preds)

                else:
                    target_boxes = xywh2xyxy(frame_gts[:, 2:6])
                    for *pred_bbox, conf in frame_preds:
                        pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1).cpu().numpy()
                        pred_bbox = xywh2xyxy(pred_bbox[:, 2:6])
                        iou = calculate_iou(pred_bbox[0, :], target_boxes)  # 计算IoU

                        best_i = np.argmax(iou)
                        if iou[best_i] > 0.5 and best_i not in correct and best_i not in detected:  # 假设IoU阈值为0.5
                            correct.append(1)
                            detected.append(best_i)
                        else:
                            correct.append(0)
                        confs.append(1)
                        clses.append(1)

                # 计算 AP
                AP, _, R, P = ap_per_class(correct, confs, clses, cls_gts)  # 假设第1列是类
                # print(f'Frame {frame}: AP = {AP.mean()}')
                mAPs.append(AP.mean())
                mR.append(R.mean())
                mP.append(P.mean())

    # 打印最终结果
    mean_mAP = np.mean(mAPs)
    mean_R = np.mean(mR)
    mean_P = np.mean(mP)

    print(f'Mean mAP: {mean_mAP}')
    print(f'Mean Recall: {mean_R}')
    print(f'Mean Precision: {mean_P}')


def load_gt(gt_folder, gt_filename):
    gt_path = os.path.join(gt_folder, gt_filename)
    gt_data = np.loadtxt(gt_path, delimiter=',')
    # 返回一个字典，键为帧号，值为（类别，边界框）
    gt_dict = {}
    for row in gt_data:
        frame_id = int(row[0])
        cls = int(row[1])  # 类别
        bbox = row[2:6]  # 边界框
        if frame_id not in gt_dict:
            gt_dict[frame_id] = []
        gt_dict[frame_id].append((cls, bbox))
    return gt_dict


def load_detections(detection_folder, detection_filename):
    # 类似于load_gt函数，读取检测数据并返回格式化数据
    detection_path = os.path.join(detection_folder, detection_filename)
    detection_data = np.loadtxt(detection_path, delimiter=',')
    detections = []
    for row in detection_data:
        frame_id = int(row[0])
        cls = int(row[1])  # 类别
        bbox = row[2:6]  # 边界框
        conf = row[6]  # 置信度
        detections.append((frame_id, cls, bbox, conf))
    return detections


def prepare_data(gt_folder, gt_filename, detection_folder, detection_filename):
    gt_dict = load_gt(gt_folder, gt_filename)
    detections = load_detections(detection_folder, detection_filename)

    targets = []
    output = []

    for frame_id in gt_dict.keys():
        if frame_id in gt_dict:
            targets.append(np.array([[cls] + list(bbox) for cls, bbox in gt_dict[frame_id]]))
        else:
            targets.append(np.empty((0, 6)))

        frame_detections = [det for det in detections if det[0] == frame_id]
        if frame_detections:
            output.append(np.array([[cls] + list(bbox) + [conf] for frame_id, cls, bbox, conf in frame_detections]))
        else:
            output.append(None)

    return targets, output


if __name__ == '__main__':
    main()
