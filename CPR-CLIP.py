import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from core.data import getDataLoader
from core.model import CLIP_img, CLIP_text
from core.text_prompt import tokenBatchLabel, tokenTestLabel
from core.clip_utils import getClipLoss, createLogits, genMatrixGT
from core.accuracy import eval_mAP_mmitmAP

import clip
import os
import argparse
from tensorboardX import SummaryWriter

def test(args, imgModel, txtModel, test_loader, logit_scale):
    imgModel.eval()
    txtModel.eval()

    scoreBCEList, scoreCLIPList, labelList = [], [], []

    txtTkn, _ = tokenTestLabel()
    txtFeat = txtModel(txtTkn) # torch.Size([78, 77]) ==> torch.Size([78, 512]) 

    for i, (feat, label) in enumerate(test_loader):
        # Predict
        imgFeat, imgScore = imgModel(feat.to('cuda'))
        logits_per_x1, _ = createLogits(imgFeat, txtFeat, logit_scale)
        # Store
        scoreBCEList.append(imgScore.detach().cpu().reshape(-1)) # imgModel prediction
        scoreCLIPList.append(logits_per_x1.detach().cpu().reshape(-1)) # CLIP prediction
        labelList.append(label.detach().cpu().reshape(-1)) 

    if args.eval_CLIP_result: # Eval
        return eval_mAP_mmitmAP(scoreCLIPList, labelList) # CLIP prediction
    else:
        return eval_mAP_mmitmAP(scoreBCEList, labelList) # imgModel prediction

def train(args):
    # #### Model Name 
    logger = SummaryWriter(log_dir=os.path.join(args.exp_path, args.exp_name))
 
    # #### Dataset loader 
    trainSet, train_loader, testSet, test_loader = getDataLoader(args.pkl_train, args.pkl_test)

    # #### Create Model 
    CLIP_base, model_state_dict = clip.load(
        name='ViT-B/16', device=args.device, jit=False, tsm=False, joint=False, T=8, dropout=0., emb_dropout=0., pretrain=True, if_proj=True)
    txtModel = CLIP_text(CLIP_base).float()
    imgModel = CLIP_img(in_channels=2048, out_channels=512, num_cls=14).cuda() 

    # #### Optimizer 
    optimizer = torch.optim.SGD(imgModel.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # #### Training 
    glbStp = 0
    for epoch in range(args.epochs):
        txtModel.train()
        imgModel.train()

        for _, (feat, label) in enumerate(train_loader):
            txtModel.zero_grad()
            imgModel.zero_grad()

            # Get prompt & token
            promptTkn, labelID = tokenBatchLabel(label)

            # Img process & Txt process 
            imgFeat, imgScore = imgModel(feat.to(args.device)) # Image feature
            txtFeat = txtModel(promptTkn) # Text feature
            
            # Loss 
            loss = 0
            logit_scale = CLIP_base.logit_scale.exp()
            if args.enable_CLIP_loss:
                matrix_GT = genMatrixGT(args, imgFeat.shape[0], labelID)
                loss_CLIP = getClipLoss(imgFeat, txtFeat, logit_scale, imgModel.KLLoss, txtModel.KLLoss, args.device, matrix_GT)
                loss += loss_CLIP
                logger.add_scalar("Loss_CLIP", loss_CLIP.item(), global_step=glbStp)
            if args.enable_BCE_loss:
                loss_BCE = imgModel.BCELoss(imgScore, label.to(args.device))
                loss += loss_BCE
                logger.add_scalar("Loss_BCE", loss_BCE.item(), global_step=glbStp)

            # Log 
            logger.add_scalar("Loss", loss.item(), global_step=glbStp) 
            logger.add_scalar("Lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step=glbStp)

            loss.backward()
            optimizer.step()

            glbStp += 1

        trainSet.initDataset()
        scheduler.step()

        print('epoch: %02d, lr: %f, loss: %f ' % (epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss), end='')
        
        # #### Save ckpt
        if (epoch + 1) % args.save_interval == 0:
            torch.save(txtModel.state_dict(), os.path.join(args.exp_path, args.exp_name, '%02d_text.pth'%(epoch + 1)))
            torch.save(imgModel.state_dict(), os.path.join(args.exp_path, args.exp_name, '%02d_img.pth'%(epoch + 1)))

        # #### Testing 
        if (epoch + 1) % args.test_interval == 0:
            mAP, mmit_mAP = test(args, imgModel, txtModel, test_loader, logit_scale)

            logger.add_scalar("mAP", mAP, global_step=epoch)
            logger.add_scalar("mmit_mAP", mmit_mAP, global_step=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    # Base settings
    parser.add_argument('--epochs', default=60, type=int, help ='Training epochs.')
    parser.add_argument('--lr', default=0.01, help ='Learning rate.')
    parser.add_argument('--test_interval', default=1, type=int, help='Testing intervals.')
    parser.add_argument('--save_interval', default=60, type=int, help='Saving intervals.')
    parser.add_argument('--device', default='cuda', type=str, help='GPU.')
    # Exp name
    parser.add_argument('--exp_name', type=str, help='Exp name.') 
    parser.add_argument('--exp_path', default='./ExpCLIP', type=str, help='Exp path.')
    # Dataset pkl
    parser.add_argument('--pkl_train', default='TSN_Single_Feat', type=str, help='pkl Train.')
    parser.add_argument('--pkl_test', default='TSN_Composite_Feat', type=str, help='pkl Test.')
    # Loss
    parser.add_argument('--enable_CLIP_loss', action='store_true', default=False, help='Add --enable_CLIP_loss to use CLIP Loss.')
    parser.add_argument('--enable_BCE_loss', action='store_true', default=False, help='Add --enable_BCE_loss to use BCE Loss.')
    # Eval
    parser.add_argument('--eval_CLIP_result', action='store_true', default=False, help='Add --eval_CLIP_result to eval CLIP results.')

    args = parser.parse_args()

    train(args)
