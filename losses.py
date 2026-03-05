import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present', ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if ignore is not None:
        mask = (labels != ignore)
        probas = probas[mask]
        labels = labels[mask]

    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(class_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = torch.mean(torch.stack([lovasz_softmax_flat(probas[i].permute(1, 2, 0).contiguous().view(-1, probas.size(1)), 
                                                           labels[i].view(-1), classes=classes, ignore=ignore) 
                                       for i in range(probas.size(0))]))
    else:
        loss = lovasz_softmax_flat(probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1)), 
                                   labels.view(-1), classes=classes, ignore=ignore)
    return loss

class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        if self.ignore_index is not None:
             valid = (labels != self.ignore_index)
             if valid.sum() == 0: return torch.tensor(0., device=logits.device, requires_grad=True)
             # Lovasz implementation needs flattening or specific handling
             # A simple way for 'masked' lovasz:
             # Just pass flattened valid pixels? 
             # The lovasz_softmax_flat function takes sparse params?
             # Let's use the 'ignore' kwarg in lovasz_softmax if supported, or masking manually
             return lovasz_softmax(probas, labels, ignore=self.ignore_index)
        return lovasz_softmax(probas, labels, ignore=self.ignore_index)
