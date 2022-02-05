import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=False, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BinaryCrossEntropyLoss
        elif mode == 'bce_with_dice':
            return self.BinaryCrossEntropyLoss_with_Dice_loss
        elif mode == 'mse':
            return self.MeanSquaredLoss
        else:
            raise NotImplementedError

    def MeanSquaredLoss(self, input, target):
        n, c, h, w = input.size()
        creiterion = nn.MSELoss(reduction='mean')
        
        if self.cuda:
            creiterion = creiterion.cuda()
        
        loss = creiterion(input, target.float())
        
        return loss

    def Dice_loss(self, input, target):
        
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

    def BinaryCrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        
        criterion = nn.BCELoss(reduction='mean')
        
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.float())

        if self.batch_average:
            loss /= n

        return loss
    
    def BinaryCrossEntropyLoss_with_Dice_loss(self, logit, target):
        n, c, h, w = logit.size()

        dice_loss = self.Dice_loss(logit, target)
        criterion = nn.BCELoss(reduction='mean')
        
        if self.cuda:
            criterion = criterion.cuda()

        loss = 0.5 * criterion(logit, target.float()) + 0.5 * dice_loss

        if self.batch_average:
            loss /= n

        return loss

    def BinaryFocalLoss(self, input, target, gamma=2):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        
        return loss.mean()
    
    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        #criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
        #                                size_average=self.size_average)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss







if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




