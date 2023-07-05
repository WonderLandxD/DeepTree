import argparse
import glob
import os
import torch
import timm
from util import AverageMeter
from tqdm import tqdm
import torch.utils.data as Dataloader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def model_opt(branch_0_model, branch_1_model, branch_2_model, branch_3_model, branch_4_model, branch_5_model, opt_loader):
    branch_0_model.eval()
    branch_1_model.eval()
    branch_2_model.eval()
    branch_3_model.eval()
    branch_4_model.eval()
    branch_5_model.eval()
    acc = AverageMeter()

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(tqdm(opt_loader, disable=False)):
            input = input.cuda()
            label = label.cuda()
            branch_0_output = branch_0_model(input)
            branch_0_pred = torch.argmax(branch_0_output, dim=1)

            if branch_0_pred.item() == 0:
                pred = torch.tensor([6]).cuda()  # Invasive

            elif branch_0_pred.item() == 1:  # ---Non-invasive

                branch_1_output = branch_1_model(input)
                branch_1_pred = torch.argmax(branch_1_output, dim=1)
                if branch_1_pred.item() == 1:  # ---Atypical or DCIS
                    # print('Atypical or DCIS')
                    branch_4_output = branch_4_model(input)
                    branch_4_pred = torch.argmax(branch_4_output, dim=1)
                    if branch_4_pred.item() == 1:
                        # print('DCIS')
                        pred = torch.tensor([5]).cuda()  # DCIS
                    elif branch_4_pred.item() == 0:  # ---Atypical
                        # print('Atypical')
                        branch_5_output = branch_5_model(input)
                        branch_5_pred = torch.argmax(branch_5_output, dim=1)
                        if branch_5_pred.item() == 0:
                            pred = torch.tensor([4]).cuda()  # ADH
                        elif branch_5_pred.item() == 1:
                            pred = torch.tensor([3]).cuda()  # FEA

                elif branch_1_pred.item() == 0:  # ---Non-atypical
                    branch_2_output = branch_2_model(input)
                    branch_2_pred = torch.argmax(branch_2_output, dim=1)
                    if branch_2_pred.item() == 0:
                        pred = torch.tensor([0]).cuda()  # Normal
                    elif branch_2_pred.item() == 1:  # ---Hyperplastic
                        branch_3_output = branch_3_model(input)
                        branch_3_pred = torch.argmax(branch_3_output, dim=1)
                        if branch_3_pred.item() == 0:
                            pred = torch.tensor([1]).cuda()  # Benign
                        elif branch_3_pred.item() == 1:
                            pred = torch.tensor([2]).cuda()  # UDH
            batch_size = label.size(0)
            acc.update(torch.sum(label == pred).item() / batch_size, batch_size)

        print('Optimizing: [{}/{}]\t'
              'acc {acc.avg:.4f}'.format(batch_idx + 1, len(opt_loader), acc=acc))

    return acc.avg


def parse_args():
    parser = argparse.ArgumentParser('Argument for optimizing')
    parser.add_argument('--gpu', default=None, help='GPU id to ues (can use multi-gpu). default=None')
    parser.add_argument('--num_select', default=0, type=int,
                        help='backbone select for trained model weights, If 0, just default name; '
                             'if NUM such as 1 or 2 or 3, using backbone_NUM. '
                             'REMEMBER: Watch Trainval dir FIRST and make decision!!!'
                             'default=0')
    parser.add_argument('--dt_select', default=None, type=str, help='res, efficient, dense, red')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print('-----------------------BRACS Datasets No1 DeepTree Optimizing (7 classes)-----------------------')

    NUM_CLASSES = 7

    OPT_ROOT = '/mnt/cpath0/wonderland/Datasets/BRACS_ROI/norm_version/NewVal'

    opt_list = glob.glob(os.path.join(OPT_ROOT, '*/*.png'))

    from data_setting import BRACSDatasets_Test

    opt_dataset = BRACSDatasets_Test(opt_list)

    opt_loader = Dataloader.DataLoader(opt_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # batch_size=1 恒定不变
    n_data = len(opt_dataset)
    print('number of validating samples (for optimizing): {}'.format(n_data))

    save_loss = './BRACS_Results'
    if not os.path.exists(save_loss):
        raise ValueError('This directory does not have the folder --- {}'.format(save_loss))

    opt_loss = os.path.join(save_loss, 'Opt_Results-NUM__{}'.format(args.num_select))
    if not os.path.exists(opt_loss):
        os.mkdir(opt_loss)
    with open(os.path.join(opt_loss, 'opt.txt'), 'w') as f:
        f.write('------Optimizing results for No.{}------\n'.format(args.num_select))

    best_acc = float('-inf')
    best_model_name = None
    if args.dt_select == 'res':
        model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    elif args.dt_select == 'efficient':
        model_list = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']
    elif args.dt_select == 'dense':
        model_list = ['densenet121', 'densenet169', 'densenet161', 'densenet201']
    else:
        model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                      'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
                      'densenet121', 'densenet169', 'densenet161', 'densenet201']
    print('model list is {}'.format(model_list))
    for model_name in model_list:
        branch_0_model = timm.create_model(model_name, pretrained=False, num_classes=2)
        branch_1_model = timm.create_model(model_name, pretrained=False, num_classes=2)
        branch_2_model = timm.create_model(model_name, pretrained=False, num_classes=2)
        branch_3_model = timm.create_model(model_name, pretrained=False, num_classes=2)
        branch_4_model = timm.create_model(model_name, pretrained=False, num_classes=2)
        branch_5_model = timm.create_model(model_name, pretrained=False, num_classes=2)

        # Multi-GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            branch_0_model = torch.nn.DataParallel(branch_0_model)
            branch_1_model = torch.nn.DataParallel(branch_1_model)
            branch_2_model = torch.nn.DataParallel(branch_2_model)
            branch_3_model = torch.nn.DataParallel(branch_3_model)
            branch_4_model = torch.nn.DataParallel(branch_4_model)
            branch_5_model = torch.nn.DataParallel(branch_5_model)

        if args.num_select == 0:
            branch_0_CKPT = torch.load(os.path.join(save_loss, 'branch0', model_name, 'acc_best_model.pth'))
            branch_1_CKPT = torch.load(os.path.join(save_loss, 'branch1', model_name, 'acc_best_model.pth'))
            branch_2_CKPT = torch.load(os.path.join(save_loss, 'branch2', model_name, 'acc_best_model.pth'))
            branch_3_CKPT = torch.load(os.path.join(save_loss, 'branch3', model_name, 'acc_best_model.pth'))
            branch_4_CKPT = torch.load(os.path.join(save_loss, 'branch4', model_name, 'acc_best_model.pth'))
            branch_5_CKPT = torch.load(os.path.join(save_loss, 'branch5', model_name, 'acc_best_model.pth'))
        else:
            branch_0_CKPT = torch.load(os.path.join(save_loss, 'branch0', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))
            branch_1_CKPT = torch.load(os.path.join(save_loss, 'branch1', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))
            branch_2_CKPT = torch.load(os.path.join(save_loss, 'branch2', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))
            branch_3_CKPT = torch.load(os.path.join(save_loss, 'branch3', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))
            branch_4_CKPT = torch.load(os.path.join(save_loss, 'branch4', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))
            branch_5_CKPT = torch.load(os.path.join(save_loss, 'branch5', '{}_{}'.format(model_name, args.num_select), 'acc_best_model.pth'))

        branch_0_model.load_state_dict(branch_0_CKPT)
        branch_1_model.load_state_dict(branch_1_CKPT)
        branch_2_model.load_state_dict(branch_2_CKPT)
        branch_3_model.load_state_dict(branch_3_CKPT)
        branch_4_model.load_state_dict(branch_4_CKPT)
        branch_5_model.load_state_dict(branch_5_CKPT)

        if torch.cuda.is_available():
            branch_0_model = branch_0_model.cuda()
            branch_1_model = branch_1_model.cuda()
            branch_2_model = branch_2_model.cuda()
            branch_3_model = branch_3_model.cuda()
            branch_4_model = branch_4_model.cuda()
            branch_5_model = branch_5_model.cuda()

        print('{} DeepTree starting......'.format(model_name))
        acc = model_opt(branch_0_model, branch_1_model, branch_2_model, branch_3_model, branch_4_model, branch_5_model, opt_loader)
        # acc = float(acc)
        print('Saving {} DeepTree accuracy results......'.format(model_name))
        with open(os.path.join(opt_loss, 'opt.txt'), 'a') as f:
            f.write('{} DeepTree\tAccuracy: {:.4f}\n'.format(model_name, acc))
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
    print('Every DeepTree has been verified......')

    name_list = []
    branch_list = []

    for i in range(NUM_CLASSES - 1):
        name_list.append(best_model_name)

        other_branch = timm.create_model(best_model_name, pretrained=False, num_classes=2)
        if args.num_select == 0:
            other_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(i), best_model_name, 'acc_best_model.pth'))
        else:
            other_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(i), '{}_{}'.format(best_model_name, args.num_select), 'acc_best_model.pth'))

        other_branch.load_state_dict(other_CKPT)
        if torch.cuda.is_available():
            other_branch = other_branch.cuda()

        branch_list.append(other_branch)


    print('Optimization using one-factor experiments......')
    with open(os.path.join(opt_loss, 'opt.txt'), 'a') as f:
        f.write('---Start Optimizing---\n')

    opt_best_acc = 0
    for branch_idx in range(NUM_CLASSES - 1):
        print('Branch {} start optimizing......'.format(branch_idx))
        opt_best_acc = best_acc
        opt_best_model_name = best_model_name
        for select_name in model_list:
            select_branch = timm.create_model(select_name, pretrained=False, num_classes=2)

            if args.num_select == 0:
                select_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(branch_idx), select_name, 'acc_best_model.pth'))
            else:
                select_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(branch_idx), '{}_{}'.format(select_name, args.num_select), 'acc_best_model.pth'))

            select_branch.load_state_dict(select_CKPT)

            if torch.cuda.is_available():
                select_branch = select_branch.cuda()

            branch_list[branch_idx] = select_branch
            name_list[branch_idx] = select_name
            print('Try {}......'.format(name_list))
            opt_acc = model_opt(branch_list[0], branch_list[1], branch_list[2], branch_list[3], branch_list[4], branch_list[5], opt_loader)
            if opt_acc >= opt_best_acc:
                opt_best_acc = opt_acc
                opt_best_model_name = select_name
                print('------ Found best solution: {}, Accuracy = {:.4f} ------'.format(name_list, opt_acc))

        opt_branch = timm.create_model(opt_best_model_name, pretrained=False, num_classes=2)
        if args.num_select == 0:
            opt_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(branch_idx), opt_best_model_name, 'acc_best_model.pth'))
        else:
            opt_CKPT = torch.load(os.path.join(save_loss, 'branch{}'.format(branch_idx), '{}_{}'.format(opt_best_model_name, args.num_select), 'acc_best_model.pth'))

        opt_branch.load_state_dict(opt_CKPT)

        if torch.cuda.is_available():
            opt_branch = opt_branch.cuda()

        branch_list[branch_idx] = opt_branch
        name_list[branch_idx] = opt_best_model_name

        with open(os.path.join(opt_loss, 'opt.txt'), 'a') as f:
            f.write('Optimized Branch {}\t{}\tAccuracy: {:.4f}\n'.format(branch_idx, name_list, opt_best_acc))

    print('------ Optimal Solution ------\n------ {} ------\n ------ Best accuracy = {:.4f} ------'.format(name_list, opt_best_acc))
    with open(os.path.join(opt_loss, 'opt.txt'), 'a') as f:
        f.write('------ Optimal Solution ------\n------ {} ------\n ------ Best accuracy = {:.4f} ------'.format(name_list, opt_best_acc))








