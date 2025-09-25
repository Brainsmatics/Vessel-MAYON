import os
import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


# ---------- 1. 数据加载 ----------
def load_3d_data(truth_dir, pred_dir):
    """加载配对的 NIfTI 文件，返回 (gt, pred) 生成器，已二值化"""
    for fname in os.listdir(truth_dir):
        if fname.endswith('.nii.gz'):
            gt   = nib.load(os.path.join(truth_dir, fname)).get_fdata()
            pred = nib.load(os.path.join(pred_dir, fname)).get_fdata()
            yield (gt > 0.5).astype(np.uint8), (pred > 0.5).astype(np.uint8)


# ---------- 2. 指标 ----------
def dice_3d(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)


def jaccard_3d(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((y_true == 0) * y_pred)
    fn = np.sum(y_true * (y_pred == 0))
    return tp / (tp + fp + fn + 1e-8)


def precision_3d(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((y_true == 0) * y_pred)
    return tp / (tp + fp + 1e-8)


def recall_3d(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (y_pred == 0))
    return tp / (tp + fn + 1e-8)


def skeleton_accuracy_3d(y_true, y_pred):
    sk_pred = skeletonize(y_pred > 0)
    intersection = np.sum(sk_pred * (y_true > 0))
    return intersection / (np.sum(sk_pred) + 1e-8)


# ---------- 3. 主入口 ----------
def calculate_all_metrics(truth_dir, pred_dir):
    metrics = {
        'Dice': [],
        'Jaccard': [],
        'Precision': [],
        'Recall': [],
        'Skeleton_Acc': []
    }

    for gt, pred in load_3d_data(truth_dir, pred_dir):
        metrics['Dice'].append(dice_3d(gt, pred))
        metrics['Jaccard'].append(jaccard_3d(gt, pred))
        metrics['Precision'].append(precision_3d(gt, pred))
        metrics['Recall'].append(recall_3d(gt, pred))
        metrics['Skeleton_Acc'].append(skeleton_accuracy_3d(gt, pred))

    return {k: np.nanmean(v) for k, v in metrics.items()}


# ---------- 4. 运行示例 ----------
if __name__ == "__main__":
    result = calculate_all_metrics(
        truth_dir=r"E:\wfr\code\nnUNet\pjy15\CAS_label15",
        pred_dir=r"E:\wfr\code\nnUNet\pjy15\result15\bffmsa4_15"
    )

    print("\n三维分割评估结果：")
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")