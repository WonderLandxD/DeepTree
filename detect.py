import argparse
import glob
import os
import torch
import timm
import numpy as np
from util import AverageMeter
from tqdm import tqdm
import torch.utils.data as Dataloader
from data_setting import BRACSDatasets_Test
from calculate_roc_f1 import New_Score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def confusion_matr_func(confusion_mat, batch_predict, batch_label):
    n1 = len(confusion_mat)
    with torch.no_grad():
        pre_tag = batch_predict
        tag = batch_label.view(-1)
        for i in range(len(tag)):
            confusion_mat[tag[i]][pre_tag[i]] += 1
    precision, recall, f1score = [], [], []
    for i in range(n1):
        rowsum, colsum = sum(confusion_mat[i]), sum(confusion_mat[r][i] for r in range(n1))
        precision.append(confusion_mat[i][i] / float(colsum))
        recall.append(confusion_mat[i][i] / float(rowsum))
        f1score.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))

    correct = [confusion_mat[i][i] for i in range(len(confusion_mat[0]))]
    total_acc = sum(correct) / sum(map(sum, confusion_mat))
    return precision, recall, f1score, total_acc, confusion_mat


def model_test(tree_0_model, tree_1_model, tree_2_model, tree_3_model,
               tree_4_model, tree_5_model, test_loader, confusion_mat):

    tree_0_model.eval()
    tree_1_model.eval()
    tree_2_model.eval()
    tree_3_model.eval()
    tree_4_model.eval()
    tree_5_model.eval()

    acc = AverageMeter()

    save_target = []
    save_pre = []

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(tqdm(test_loader, disable=False)):

            input = input.cuda()
            label = label.cuda()

            tree_0_output = tree_0_model(input)
            tree_0_pred = torch.argmax(tree_0_output, dim=1)

            if tree_0_pred.item() == 0:
                pred = torch.tensor([6]).cuda() # Invasive

            elif tree_0_pred.item() == 1:  # ---Non-invasive

                tree_1_output = tree_1_model(input)
                tree_1_pred = torch.argmax(tree_1_output, dim=1)
                if tree_1_pred.item() == 1: # ---Atypical or DCIS
                    # print('Atypical or DCIS')
                    tree_4_output = tree_4_model(input)
                    tree_4_pred = torch.argmax(tree_4_output, dim=1)
                    if tree_4_pred.item() == 1:
                        # print('DCIS')
                        pred = torch.tensor([5]).cuda() # DCIS
                    elif tree_4_pred.item() == 0: # ---Atypical
                        # print('Atypical')
                        tree_5_output = tree_5_model(input)
                        tree_5_pred = torch.argmax(tree_5_output, dim=1)
                        if tree_5_pred.item() == 0:
                            pred = torch.tensor([4]).cuda() # ADH
                        elif tree_5_pred.item() == 1:
                            pred = torch.tensor([3]).cuda() # FEA

                elif tree_1_pred.item() == 0:   # ---Non-atypical
                    tree_2_output = tree_2_model(input)
                    tree_2_pred = torch.argmax(tree_2_output, dim=1)
                    if tree_2_pred.item() == 0:
                        pred = torch.tensor([0]).cuda()  # Normal
                    elif tree_2_pred.item() == 1:  # ---Hyperplastic
                        tree_3_output = tree_3_model(input)
                        tree_3_pred = torch.argmax(tree_3_output, dim=1)
                        if tree_3_pred.item() == 0:
                            pred = torch.tensor([1]).cuda() # Benign
                        elif tree_3_pred.item() == 1:
                            pred = torch.tensor([2]).cuda() # UDH

            batch_size = label.size(0)
            acc.update(torch.sum(label == pred).item() / batch_size, batch_size)


            _, _, _, _, confusion_mat = confusion_matr_func(confusion_mat, pred, label)

            save_target.extend(list(label.cpu().numpy()))
            save_pre.extend(list(pred.cpu().numpy()))

        print('Test: [{}/{}]\t'
              'acc {acc.val:.3f} ({acc.avg:.3f})'.format(batch_idx + 1, len(test_loader), acc=acc))

    return save_target, save_pre, confusion_mat


def parse_args():
    parser = argparse.ArgumentParser('Argument for testing')

    parser.add_argument('--gpu', default='0', help='GPU id to ues (can use multi-gpu). default=0')
    parser.add_argument('--backbone0', default=None, help='backbone of CNN tree 0. default=None')
    parser.add_argument('--model0_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--backbone1', default=None, help='backbone of CNN tree 1. default=None')
    parser.add_argument('--model1_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--backbone2', default=None, help='backbone of CNN tree 2. default=None')
    parser.add_argument('--model2_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--backbone3', default=None, help='backbone of CNN tree 3. default=None')
    parser.add_argument('--model3_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--backbone4', default=None, help='backbone of CNN tree 4. default=None')
    parser.add_argument('--model4_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--backbone5', default=None, help='backbone of CNN tree 5. default=None')
    parser.add_argument('--model5_select', default='acc', help='model select for tree 0. --acc or loss default=acc')
    parser.add_argument('--num_select', default=0, type=int, help='backbone select for trained model weights, If 0, just default name; if NUM such as 1 or 2 or 3, using backbone_NUM. REMEMBER: Watch Trainval dir FIRST and make decision!!!')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('-----------------------BRACS Datasets No1 DeepTree validating (7 classes)-----------------------')


    NUM_CLASSES = 7

    TEST_ROOT = '/mnt/cpath0/wonderland/Datasets/BRACS_ROI/norm_version/NewVal'

    test_list = glob.glob(os.path.join(TEST_ROOT, '*/*.png'))

    test_dataset = BRACSDatasets_Test(test_list)

    test_loader = Dataloader.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                       pin_memory=True) # batch_size=1 恒定不变

    n_data = len(test_dataset)
    print('number of testing samples: {}'.format(n_data))

    save_loss = './BRACS_Results'
    if not os.path.exists(save_loss):
        raise ValueError('This directory does not have the folder --- {}'.format(save_loss))

    tree_0_model = timm.create_model(args.backbone0, pretrained=False, num_classes=2)  # I vs N+B+A+U+F+D
    tree_1_model = timm.create_model(args.backbone1, pretrained=False, num_classes=2)  # N+B+U vs A+F+D
    tree_2_model = timm.create_model(args.backbone2, pretrained=False, num_classes=2)  # N vs B+U
    tree_3_model = timm.create_model(args.backbone3, pretrained=False, num_classes=2)  # B vs U
    tree_4_model = timm.create_model(args.backbone4, pretrained=False, num_classes=2)  # A+F vs D
    tree_5_model = timm.create_model(args.backbone5, pretrained=False, num_classes=2)  # A vs F

    print('---branch0: {} ---branch1: {} ---branch2: {} ---branch3: {} ---branch4: {} ---branch5: {} '.format(
        args.backbone0, args.backbone1, args.backbone2, args.backbone3, args.backbone4, args.backbone5))

    # Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        tree_0_model = torch.nn.DataParallel(tree_0_model)
        tree_1_model = torch.nn.DataParallel(tree_1_model)
        tree_2_model = torch.nn.DataParallel(tree_2_model)
        tree_3_model = torch.nn.DataParallel(tree_3_model)
        tree_4_model = torch.nn.DataParallel(tree_4_model)
        tree_5_model = torch.nn.DataParallel(tree_5_model)

    if args.num_select == 0:
        tree_0_CKPT = torch.load(os.path.join(save_loss, 'branch0', args.backbone0, '{}_best_model.pth'.format(args.model0_select)))
        tree_1_CKPT = torch.load(os.path.join(save_loss, 'branch1', args.backbone1, '{}_best_model.pth'.format(args.model1_select)))
        tree_2_CKPT = torch.load(os.path.join(save_loss, 'branch2', args.backbone2, '{}_best_model.pth'.format(args.model2_select)))
        tree_3_CKPT = torch.load(os.path.join(save_loss, 'branch3', args.backbone3, '{}_best_model.pth'.format(args.model3_select)))
        tree_4_CKPT = torch.load(os.path.join(save_loss, 'branch4', args.backbone4, '{}_best_model.pth'.format(args.model4_select)))
        tree_5_CKPT = torch.load(os.path.join(save_loss, 'branch5', args.backbone5, '{}_best_model.pth'.format(args.model5_select)))

    else:
        tree_0_CKPT = torch.load(os.path.join(save_loss, 'branch0', '{}_{}'.format(args.backbone0, args.num_select), '{}_best_model.pth'.format(args.model0_select)))
        tree_1_CKPT = torch.load(os.path.join(save_loss, 'branch1', '{}_{}'.format(args.backbone1, args.num_select), '{}_best_model.pth'.format(args.model1_select)))
        tree_2_CKPT = torch.load(os.path.join(save_loss, 'branch2', '{}_{}'.format(args.backbone2, args.num_select), '{}_best_model.pth'.format(args.model2_select)))
        tree_3_CKPT = torch.load(os.path.join(save_loss, 'branch3', '{}_{}'.format(args.backbone3, args.num_select), '{}_best_model.pth'.format(args.model3_select)))
        tree_4_CKPT = torch.load(os.path.join(save_loss, 'branch4', '{}_{}'.format(args.backbone4, args.num_select), '{}_best_model.pth'.format(args.model4_select)))
        tree_5_CKPT = torch.load(os.path.join(save_loss, 'branch5', '{}_{}'.format(args.backbone5, args.num_select), '{}_best_model.pth'.format(args.model5_select)))

    tree_0_model.load_state_dict(tree_0_CKPT)
    tree_1_model.load_state_dict(tree_1_CKPT)
    tree_2_model.load_state_dict(tree_2_CKPT)
    tree_3_model.load_state_dict(tree_3_CKPT)
    tree_4_model.load_state_dict(tree_4_CKPT)
    tree_5_model.load_state_dict(tree_5_CKPT)

    if torch.cuda.is_available():
        tree_0_model = tree_0_model.cuda()
        tree_1_model = tree_1_model.cuda()
        tree_2_model = tree_2_model.cuda()
        tree_3_model = tree_3_model.cuda()
        tree_4_model = tree_4_model.cuda()
        tree_5_model = tree_5_model.cuda()

    print("==> testing...")
    confusion_mat = np.array([[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)])
    ft_target, ft_pre, confusion_mat = model_test(tree_0_model, tree_1_model, tree_2_model, tree_3_model,
                                                             tree_4_model, tree_5_model, test_loader, confusion_mat)

    classes = ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC']

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(
        include_values=True,
        cmap='plasma',
        ax=None,
        xticks_rotation='horizontal',
        values_format='d'
    )

    test_loss = os.path.join(save_loss, 'TEST/exp')
    if not os.path.exists(test_loss):
        os.makedirs(test_loss)
    else:
        for i in range(1, 100):
            test_loss = os.path.join(save_loss, 'TEST/exp{}'.format(i))
            if os.path.exists(test_loss):
                i += 1
            else:
                os.makedirs(test_loss)
                break

    plt.savefig(os.path.join(test_loss, 'confusion_matrix.png'))

    with open(os.path.join(test_loss, 'BRACS_test_results.csv'), 'w') as f:
        f.write('{}, {}, {}, {}, {}, {}\n'.format(args.backbone0, args.backbone1, args.backbone2, args.backbone3, args.backbone4, args.backbone5))
        f.write('acc, precision, recall, f1, roc auc\n')

    score = New_Score(ft_target, ft_pre)
    acc = score.cal_acc()
    precision = score.cal_precision()
    recall = score.cal_recall()
    f1 = score.cal_f1()

    with open(os.path.join(test_loss, 'BRACS_test_results.csv'), 'a') as f:
        f.write('%0.6f,%0.6f,%0.6f,%0.6f,\n' % (acc, precision, recall, f1))

    # np.save(os.path.join(test_loss, 'fpr_roc.npy'), fpr)
    # np.save(os.path.join(test_loss, 'tpr_roc.npy'), tpr)
    print('ALL is Done')

