import numpy as np
import warnings
import cv2

def get_miou_prec_recall_fscore(total_tp, total_fp, total_fn):
    miou = total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)
    return miou.item(), prec.item(), recall.item(), fscore.item()

class my_PD_FA(object):
    def __init__(self, ):
        self.reset()

    def update(self, pred, label):
        pred = np.squeeze(pred)
        label = np.squeeze(label)

        if pred.ndim == 3:
            warnings.warn(f"Pred is 3D ({pred.shape}), taking 1st channel to force 2D.")
            pred = pred[..., 0]
        if label.ndim == 3:
            warnings.warn(f"Label is 3D ({label.shape}), taking 1st channel to force 2D.")
            label = label[..., 0]

        if pred.ndim != 2 or label.ndim != 2:
            warnings.warn(f"Failed to force 2D. Pred: {pred.ndim}D, Label: {label.ndim}D. Skipping.")
            return

        max_pred = np.max(pred)
        if max_pred > 0:
            pred = pred / max_pred
        pred_binary = (pred > 0.5).astype(np.bool_)
        label = label.astype(np.uint8)

        try:
            num_labels, labels_cc, _, centroids = cv2.connectedComponentsWithStats(label, connectivity=8)
        except cv2.error as e:
            warnings.warn(f"OpenCV error: {str(e)}. Skipping.")
            return

        if num_labels <= 1:
            self.target_nums += 0
        else:
            back_mask = labels_cc == 0
            self.background_area += np.sum(back_mask)
            self.target_nums += (num_labels - 1)  

            tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
            self.false_detect += tmp_false_detect

     
            for target_id in range(1, num_labels):
                target_mask = labels_cc == target_id
                has_true = np.sum(np.logical_and(target_mask, pred_binary)) > 0
                self.true_detect += 1 if has_true else 0

    def get(self):
        
        Pd = self.true_detect / (np.spacing(1) + self.target_nums)
    
        Fa = self.false_detect / (np.spacing(1) + self.background_area)
        return Pd, Fa 

    def calculate_PD_FA(self):
        return self.get()

    def reset(self):
        self.false_detect = 0   
        self.true_detect = 0    
        self.background_area = 0 
        self.target_nums = 0    