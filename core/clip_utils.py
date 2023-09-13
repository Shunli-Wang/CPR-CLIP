import torch
import torch.nn.functional as F
import torch.nn as nn

def createLogits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True) # [8, 512]
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t() # [8, 8]
    logits_per_x2 = logit_scale * x2 @ x1.t() # [8, 8]

    # shape = [global_batch_size, global_batch_size] 
    return logits_per_x1, logits_per_x2

def getClipLoss(imgEmb, txtEmb, logit_scale, loss_img, loss_txt, device, labels):
    # imgEmb: [8, 512] 
    # txtEmb: [8, 512] 
    logits_per_image, logits_per_text = createLogits(imgEmb, txtEmb, logit_scale) # cosine similarity 

    ground_truth = torch.tensor(labels, dtype=imgEmb.dtype, device=device)

    loss_imgs = loss_img(logits_per_image, ground_truth)
    loss_texts = loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_imgs + loss_texts) / 2 
    return total_loss

def genMatrixGT(args, matSize, labelID):
    '''Get the CLIP GT Matrix.'''
    labels = torch.eye(matSize, dtype=torch.float, device=args.device, requires_grad=False)
    for j in range(0, len(labelID)): # for equal items 
        for k in range(j+1, len(labelID)):
            if torch.tensor(labelID[j]).equal(torch.tensor(labelID[k])):
                labels[j][k] = 1.0
                labels[k][j] = 1.0
    return labels

class KLLoss(nn.Module):

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, dim=1)
        probs2 = F.softmax(label * 10, dim=1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
