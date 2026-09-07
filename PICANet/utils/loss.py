import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self, smooth=1e-4, weight=None):
        super(SoftLoULoss, self).__init__()
        self.smooth = smooth  
        self.weight = weight  

    def forward(self, pred, target):
        batch_size = pred.size(0)

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3)) 
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))  

        iou = (intersection_sum + self.smooth) / (pred_sum + target_sum - intersection_sum + self.smooth)

        if self.weight is not None:
            target_ratio = target_sum / (target.size(2)*target.size(3))  
            sample_weight = torch.where(target_ratio < 0.01, self.weight, 1.0).to(pred.device)
            iou = iou * sample_weight
        
        loss = 1 - torch.mean(iou)
        return loss


class EdgeAwareSoftIoULoss(nn.Module):
    def __init__(self, smooth=1e-4, edge_weight=2.0):
        super(EdgeAwareSoftIoULoss, self).__init__()
        self.smooth = smooth
        self.edge_weight = edge_weight
        self.edge_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        base_iou = (intersection_sum + self.smooth) / (pred_sum + target_sum - intersection_sum + self.smooth)

        edge_kernel = self.edge_kernel.repeat(target.size(1), 1, 1, 1).to(pred.device)
        target_edge = F.conv2d(target.float(), edge_kernel, padding=1)
        target_edge = (target_edge > 0).float() 

        edge_intersection = pred * target_edge
        edge_intersection_sum = torch.sum(edge_intersection, dim=(1,2,3))
        edge_target_sum = torch.sum(target_edge, dim=(1,2,3))
        edge_iou = (edge_intersection_sum + self.smooth) / (torch.sum(pred*target_edge, dim=(1,2,3)) + edge_target_sum - edge_intersection_sum + self.smooth)

        total_iou = (base_iou + self.edge_weight * edge_iou) / (1 + self.edge_weight)
        loss = 1 - torch.mean(total_iou)
        return loss