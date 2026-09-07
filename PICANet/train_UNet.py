
import os
import os.path as osp
import time
import datetime
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data as Data
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import numpy as np
import cv2

from models import get_model
from utils.data import *
from modules.RSU import _upsample_like
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.logger import setup_logger

def parse_args():
    parser = ArgumentParser(description='Training PICA_UNet')
    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='nudt')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr-scheduler', type=str, default='poly')
    parser.add_argument('--net-name', type=str, default='pica_unet')
    parser.add_argument('--save-iter-step', type=int, default=1)
    parser.add_argument('--log-per-iter', type=int, default=10)
    parser.add_argument('--base-dir', type=str, default='./train_result/')
    parser.add_argument('--cp1_weight', type=float, default=1.0)
    parser.add_argument('--cp2_weight', type=float, default=0.5)

    args = parser.parse_args()

    args.time_name = time.strftime('%Y%m%dT%H-%M-%S', time.localtime())
    args.folder_name = f"{args.time_name}_{args.net_name}_{args.dataset}"
    args.save_folder = osp.join(args.base_dir, args.folder_name)
    os.makedirs(args.save_folder, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.logger = setup_logger("pica_UNet", args.save_folder, 0, filename='log.txt')
    return args

def generate_soft_prior(gt_tensor, sigma=2):
    soft_prior_list = []
    gt_np = gt_tensor.cpu().numpy()
    for b in range(gt_np.shape[0]):
        gt_single = gt_np[b, 0]
        if gt_single.max() > 0:
            blurred = cv2.GaussianBlur(gt_single, ksize=(5,5), sigmaX=sigma)
            blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
        else:
            blurred = np.zeros_like(gt_single)
        soft_prior_list.append(blurred[np.newaxis, ...])
    return torch.tensor(np.stack(soft_prior_list, axis=0), dtype=torch.float32).to(gt_tensor.device)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        if args.dataset == 'nudt':
            self.base_data_dir = './datasets/NUDT-SIRST'
        elif args.dataset == 'irstd1k':
            self.base_data_dir = './datasets/IRSTD-1k'
        elif args.dataset == 'sirstaug' or args.dataset == 'sirst':
            self.base_data_dir = './datasets/SIRST'
        else:
            raise NotImplementedError

        self.trainset = NUDTSIRSTDataset(base_dir=self.base_data_dir, mode='train', base_size=args.base_size)
        self.valset = NUDTSIRSTDataset(base_dir=self.base_data_dir, mode='test', base_size=args.base_size)
        
        self.train_data_loader = Data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(self.valset, batch_size=args.batch_size, shuffle=False)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = get_model(args.net_name).to(self.device)

        self.softiou = SoftLoULoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.cp1_loss_fn = nn.BCEWithLogitsLoss()
        self.cp2_loss_fn = nn.MSELoss()
        
        self.channel_proj = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0).to(self.device)

        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)

        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0

        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.logger = args.logger
        self.logger.info(args)

    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels):
        loss0 = self.bce_loss(d0, labels)
        loss1 = self.bce_loss(d1, labels)
        loss2 = self.bce_loss(d2, labels)
        loss3 = self.bce_loss(d3, labels)
        loss4 = self.bce_loss(d4, labels)
        loss5 = self.bce_loss(d5, labels)
        loss6 = self.bce_loss(d6, labels)
        total_loss_bce = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss0, total_loss_bce
        
    def training(self):
        start_time = time.time()
        base_log = ("Epoch: [{:d}/{:d}] -[{:d}/{:d}]|| LR: {:.6f} || BCE: {:.4f} || IoU: {:.4f} || CP2: {:.4f} || Total: {:.4f} || Cost Time: {} || ETA: {}")
        
        for epoch in range(self.args.epochs):
            self.net.train()
            
            for i, (data, labels) in enumerate(self.train_data_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.scheduler(self.optimizer, i, epoch, self.best_miou)

                # Forward
                d0, d1, d2, d3, d4, d5, d6, cp1, cp2, hx6_with_cp2 = self.net(data) 
                pred = torch.sigmoid(d0)

                # Losses
                loss0, loss_bce = self.muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
                loss_iou = self.softiou(pred, labels)
                
                cp1_soft_gt = generate_soft_prior(labels)
                cp1_loss = self.cp1_loss_fn(cp1, cp1_soft_gt) * self.args.cp1_weight
                
                cp2_ref = _upsample_like(hx6_with_cp2, cp2)
                cp2_ref = F.normalize(cp2_ref, p=2, dim=1)
                cp2_ref = self.channel_proj(cp2_ref)
                cp2_loss = self.cp2_loss_fn(cp2, cp2_ref) * self.args.cp2_weight
                
                loss_cp = cp2_loss
                
                total_loss = 10.0 * loss_iou + 10.0 * loss_bce + 10.0 * loss_cp
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                self.iter_num += 1
                
                if self.iter_num % self.args.log_per_iter == 0:
                    cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                    eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                    self.logger.info(base_log.format(
                        epoch + 1, self.args.epochs, i + 1, self.iter_per_epoch, 
                        float(self.optimizer.param_groups[0]['lr']),
                        float(loss_bce), float(loss_iou), float(cp2_loss), float(total_loss), 
                        cost_string, eta_string))

            # Validation call after each epoch
            if (epoch + 1) % self.args.save_iter_step == 0:
                self.validation()

    def validation(self):
        self.metric.reset()
        base_log = "Validation - mIoU: {:.4f} (best {:.4f}) | F1: {:.4f} (best {:.4f})"
        self.net.eval()
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                data, labels = data.to(self.device), labels.to(self.device)
                pred = self.net(data)
                self.metric.update(pred.cpu(), labels.cpu())

        miou, prec, recall, fmeasure = self.metric.get()
        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'latest.pkl'))

        if miou > self.best_miou:
            self.best_miou = miou
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, 'best.pkl'))
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        self.writer.add_scalar('Val/mIoU', miou, self.iter_num)
        self.writer.add_scalar('Val/F1', fmeasure, self.iter_num)
        self.logger.info(base_log.format(miou, self.best_miou, fmeasure, self.best_fmeasure))

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.training()
    print(f'Best mIoU: {trainer.best_miou:.5f}, Best F1: {trainer.best_fmeasure:.5f}')
