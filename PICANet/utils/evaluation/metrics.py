import torch
import torch.nn.functional as F
import numpy as np

class SigmoidMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)
        tp, fp, fn = self.batch_tp_fp_fn(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        # F1 score
        F1 = 2.0 * self.total_tp / (2.0 * self.total_tp + self.total_fp + self.total_fn + np.spacing(1))
        return pixAcc, mIoU, F1

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.detach().numpy()
        target = target.detach().numpy()

        predict = (output > 0.22).astype('int64')  # threshold
        pixel_labeled = np.sum(target > 0)
        pixel_correct = np.sum((predict == target) * (target > 0))
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1
        nbins = 1
        predict = (output.detach().numpy() > 0.22).astype('int64')
        target = target.numpy().astype('int64')
        intersection = predict * (predict == target)

        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        return area_inter, area_union

    def batch_tp_fp_fn(self, output, target):
        predict = (output.detach().numpy() > 0.22).astype('int64')
        target = target.detach().numpy().astype('int64')
        tp = np.sum((predict == 1) & (target == 1))
        fp = np.sum((predict == 1) & (target == 0))
        fn = np.sum((predict == 0) & (target == 1))
        return tp, fp, fn


class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        inter_arr, union_arr, tp_arr, fp_arr, fn_arr = self.batch_intersection_union(preds, labels,
                                                                                    self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)
        self.total_tp = np.append(self.total_tp, tp_arr)
        self.total_fp = np.append(self.total_fp, fp_arr)
        self.total_fn = np.append(self.total_fn, fn_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        F1 = 2.0 * self.total_tp.sum() / (2.0 * self.total_tp.sum() + self.total_fp.sum() + self.total_fn.sum() + np.spacing(1))
        return IoU, mIoU, F1

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_tp = np.array([])
        self.total_fp = np.array([])
        self.total_fn = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        mini = 1
        maxi = 1
        nbins = 1

        predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64')
        target = target.detach().numpy().astype('int64')
        intersection = predict * (predict == target)

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)
        tp_arr = np.zeros(num_sample)
        fp_arr = np.zeros(num_sample)
        fn_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            # TP, FP, FN
            tp_arr[b] = np.sum((predict[b] == 1) & (target[b] == 1))
            fp_arr[b] = np.sum((predict[b] == 1) & (target[b] == 0))
            fn_arr[b] = np.sum((predict[b] == 0) & (target[b] == 1))

        return area_inter_arr, area_union_arr, tp_arr, fp_arr, fn_arr
