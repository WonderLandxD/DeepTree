import argparse
import glob
import os
import torch
import time
from tqdm import tqdm
import torch.utils.data as Dataloader
import torch.nn as nn
import torch.optim as optim
from util import AverageMeter
import timm
from calculate_roc_f1 import New_Score


def model_train(model, train_loader, criterion, optimizer, epoch, args):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, disable=False)):
        optimizer.zero_grad()
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)
        preds = torch.argmax(outputs, dim=1)

        acc.update(torch.sum(labels == preds).item() / batch_size, batch_size)

        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics and write
    print('BRACS_Branch_{}_Backbone:{}'.format(args.branch, args.backbone))
    print('Train: [{0}][{1}/{2}]\t'
          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'loss {loss.val:.3f} ({loss.avg:.3f})\t'
          'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
        epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, loss=losses,
        acc=acc))

    return losses.avg, acc.avg


def model_val(model, val_loader, criterion, epoch, args):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    save_target = []
    save_pre = []

    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, disable=False)):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            preds = torch.argmax(outputs, dim=1)

            acc.update(torch.sum(labels == preds).item() / batch_size, batch_size)

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            save_target.extend(list(labels.cpu().numpy()))
            save_pre.extend(list(preds.cpu().numpy()))

            # print statistics and write
        print('BRACS_Branch_{}_Backbone:{}'.format(args.branch, args.backbone))
        print('Val: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})\t'
              'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch, batch_idx + 1, len(val_loader), batch_time=batch_time, loss=losses,
            acc=acc))

    return losses.avg, acc.avg, save_target, save_pre


def parse_args():

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--gpu', default='0', help='GPU id to ues, can use multi-gpu, default=0')
    parser.add_argument('--num_epoch', type=int, default=100, help='epochs to train for, default=100')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size,  default=32')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate, default=0.0001.')
    parser.add_argument('--backbone', default=None, type=str, help='type of backbone, default=None')   # swin_base_patch4_window7_224_in22k
    parser.add_argument('--pretrained', default=True, type=bool, help='True: Using ImageNet pretrained CNN model, '
                                                                      'False: Do not use. default=True')
    parser.add_argument('--branch', default=100, type=int, help='decision tree '
                                                              '- 0: I vs N+B+A+U+F+D, 1: N+B+U vs A+F+D, '
                                                              '2: N vs B+U, 3: B vs U, 4: A+F vs D, 5: A vs F')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate default=0.2')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('-----------------------BRACS Datasets No1 DeepTree Training (7 classes)-----------------------')

    TRAIN_ROOT = '/mnt/cpath0/wonderland/Datasets/BRACS_ROI/norm_version/train'
    VALID_ROOT = '/mnt/cpath0/wonderland/Datasets/BRACS_ROI/norm_version/NewVal'

    if args.branch == 0:
        print('----------Using branch 0: I vs N+B+A+U+F+D----------')
        CLASSES_NAME = ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_I_vs_NBAUFD

        train_dataset = BRACSDatasets_I_vs_NBAUFD(train_list)
        valid_dataset = BRACSDatasets_I_vs_NBAUFD(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch0'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    elif args.branch == 1:
        print('----------Using branch 1: N+B+U vs A+F+D----------')
        CLASSES_NAME = ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_NBU_vs_AFD

        train_dataset = BRACSDatasets_NBU_vs_AFD(train_list)
        valid_dataset = BRACSDatasets_NBU_vs_AFD(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch1'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    elif args.branch == 2:
        print('----------Using branch 2: N vs B+U----------')
        CLASSES_NAME = ['0_N', '1_PB', '2_UDH']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_N_vs_BU

        train_dataset = BRACSDatasets_N_vs_BU(train_list)
        valid_dataset = BRACSDatasets_N_vs_BU(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch2'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    elif args.branch == 3:
        print('----------Using branch 3: B vs U----------')
        CLASSES_NAME = ['1_PB', '2_UDH']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_B_vs_U

        train_dataset = BRACSDatasets_B_vs_U(train_list)
        valid_dataset = BRACSDatasets_B_vs_U(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch3'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    elif args.branch == 4:  # A+F vs D
        print('----------Using branch 4: A+F vs D----------')
        CLASSES_NAME = ['3_FEA', '4_ADH', '5_DCIS']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_AF_vs_D

        train_dataset = BRACSDatasets_AF_vs_D(train_list)
        valid_dataset = BRACSDatasets_AF_vs_D(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch4'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    elif args.branch == 5:   #A vs F
        print('----------Using branch 5: A vs F----------')
        CLASSES_NAME = ['3_FEA', '4_ADH']
        train_list, val_list = [], []
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(TRAIN_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                train_list.append(temp)
        for CLASS in CLASSES_NAME:
            temp_list = glob.glob(os.path.join(VALID_ROOT, CLASS, '*.png'))
            for temp in temp_list:
                val_list.append(temp)

        from data_setting import BRACSDatasets_A_vs_F

        train_dataset = BRACSDatasets_A_vs_F(train_list)
        valid_dataset = BRACSDatasets_A_vs_F(val_list)

        train_loader = Dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True)
        val_loader = Dataloader.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)

        ### Save Log ###
        save_loss = './BRACS_Results/branch5'
        if not os.path.exists(save_loss):
            os.makedirs(save_loss)

    else:
        raise ValueError('You need to select one of decision tree. Using --branch <int>  '
                         '- 0: I vs N+B+A+U+F+D, 1: N+B+U vs A+F+D, 2: N vs B+U, 3: B vs U, 4: A+F vs D, 5: A vs F')

    n_data = len(train_dataset)
    print('---number of training samples: {}---'.format(n_data))
    n_data = len(valid_dataset)
    print('---number of validation samples: {}---'.format(n_data))


    model = timm.create_model(args.backbone, pretrained=args.pretrained, num_classes=2, drop_rate=args.dropout)
    print('---Using {} Backbone---'.format(args.backbone))

    backbone_loss = os.path.join(save_loss, '{}'.format(args.backbone))
    if not os.path.exists(backbone_loss):
        os.makedirs(backbone_loss)

    else:
        for i in range(1, 100):
            backbone_loss = os.path.join(save_loss, '{}_{}'.format(args.backbone, i))
            if os.path.exists(backbone_loss):
                i += 1
            else:
                os.makedirs(backbone_loss)
                break

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    CrossEntropy_criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropy_criterion.cuda()

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.005)
    print('--Using NEW optimizer: optim.Adam(model.parameters(), lr={}, betas=(0.5, 0.9), weight_decay=0.005)--'.format(args.lr))

    # Training Model
    start_epoch = 1
    prev_best_val_loss = float('inf')
    prev_best_val_acc = float('-inf')
    # prev_best_val_f1 = float('-inf')

    with open(os.path.join(backbone_loss, '{}_train_results.csv'.format(args.backbone)), 'w') as f:
        f.write('epoch, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1\n')

    EPOCH = args.num_epoch

    for epoch in range(start_epoch, EPOCH + 1):

        print("==> training...")

        time_start = time.time()

        train_losses, train_acc = model_train(model, train_loader, criterion, optimizer, epoch, args)
        print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

        print("==> validation...")
        val_losses, val_acc, ft_target, ft_pre = model_val(model, val_loader, criterion, epoch, args)

        score = New_Score(ft_target, ft_pre)
        acc = score.cal_acc()
        val_precision = score.cal_precision()
        val_recall = score.cal_recall()
        val_f1 = score.cal_f1()

        # Log results
        with open(os.path.join(backbone_loss, '{}_train_results.csv'.format(args.backbone)), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (epoch, train_losses, train_acc, val_losses, val_acc, val_precision, val_recall, val_f1))

        if (val_acc >= prev_best_val_acc):
            print('==> best acc Saving...')
            torch.save(model.state_dict(), '{}/acc_best_model.pth'.format(backbone_loss))
            prev_best_val_acc = val_acc

        if (val_losses <= prev_best_val_loss):
            print('==> best loss Saving ...')
            torch.save(model.state_dict(), '{}/loss_best_model.pth'.format(backbone_loss))
            prev_best_val_loss = val_losses
