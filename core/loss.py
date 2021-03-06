import torch
from torch import nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target, mask):
        loss = self.l1_loss(target, pred)
        loss = loss * mask
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='none'):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = torch.nn.BCELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        loss = self.__loss(y_pred, y_true)
        loss *= self.__alpha * torch.pow(
            torch.abs(y_pred - y_true), self.__gamma)
        loss = torch.sum(loss, 1)
        return loss


# class ArcFaceLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=0.25, reduction='none'): # epsilon=1e-7, 
#         super(ArcFaceLoss, self).__init__()
#         self.__gamma = gamma
#         self.__alpha = alpha
#         # self.epsilon = epsilon
#         # self.__loss = nn.CrossEntropyLoss(reduction=reduction)
#         self.__loss = torch.nn.BCELoss(reduction=reduction)

#     def forward(self, y_pred, y_true, mask):
#         loss = self.__loss(y_pred, y_true)
#         loss *= self.__alpha * torch.pow(
#             torch.abs(y_pred - y_true), self.__gamma)
#         loss = torch.sum(loss, 1)
#         loss = loss * mask
#         return loss

class ArcFaceLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(ArcFaceLoss, self).__init__()
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_true, mask):
        bs, c, w, h = y_pred.shape
        mask = torch.reshape(mask, (bs, w * h))
        # y_true = y_true.permute(0, 2, 3, 1) * mask
        y_true = torch.reshape(y_true, (bs, c, w * h))
        # y_pred = y_pred.permute(0, 2, 3, 1) * mask
        y_pred = torch.reshape(y_pred, (bs, c, w * h))
        y_true = torch.argmax(y_true, 1)#.type(torch.float32)
        loss = self.cross_entropy(y_pred, y_true) 
        loss = loss * mask
        return loss

class TotalLoss(torch.nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.heat_map_lambda = 0 # 1
        self.offset_lambda = 0 # 1
        self.wh_lambda = 0 #5
        self.id_lambda = 1

        self.l1_loss = L1Loss()
        self.focal_loss = FocalLoss()
        self.id_loss = ArcFaceLoss()
        
    def forward(self, heat_map, offset, wh, id_info, heat_map_pred, offset_pred, wh_pred, id_pred):
        batch_size, _, _, _ = heat_map.shape
        loss_mask = torch.argmax(heat_map, 1)
        loss_mask = torch.clip(loss_mask, 0, 1)
        heat_map_loss = self.focal_loss(heat_map_pred, heat_map)
        offset_loss = self.l1_loss(offset_pred, offset, loss_mask)
        wh_loss = self.l1_loss(wh_pred, wh, loss_mask)
        id_loss = self.id_loss(id_pred, id_info, loss_mask)
        heat_map_loss = (torch.sum(heat_map_loss) * self.heat_map_lambda) / batch_size
        offset_loss = (torch.sum(offset_loss) * self.offset_lambda) / batch_size
        wh_loss = (torch.sum(wh_loss) * self.wh_lambda) / batch_size
        id_loss = (torch.sum(id_loss) * self.id_lambda) / batch_size
        return heat_map_loss + offset_loss + wh_loss + id_loss, [heat_map_loss, offset_loss, wh_loss, id_loss]