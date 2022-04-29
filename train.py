import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/content/TransUNet/Data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/content/TransUNet/lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--GoogleUse', type=bool,
                     default=False, help='choose the type you want to train')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True    
    data_choose=int(input('选择要训练的数据集1.Synapse 2.Corona：'))
    if data_choose ==1:
      args.dataset='Synapse'
    elif data_choose ==2:
      args.dataset ='Corona'
      
    vit_choose=int(input('请选择你想要训练的模型 1.R50-vit-b_16  2.vit-b_16  3.vit-b_32  4.vit-l_16 5.R152-vit-b-16  6.R50-vit-l-16:  '))
    flag32=False
    if vit_choose == 2:
      args.vit_name='ViT-B_16'
    elif vit_choose == 3:
      args.vit_name='ViT-B_32'
      args.vit_patches_size=32
      flag32=True
    elif vit_choose == 4:
      args.vit_name='ViT-L_16'
      args.batch_size=20
    elif vit_choose == 5:
      args.vit_name='R152-ViT-B_16'
      args.max_epochs=40
      args.n_skip=3
    elif vit_choose == 6:
      args.vit_name='R50-ViT-L_16'
      args.n_skip=3
    else:
      args.n_skip=3
    print('您选择的是：',args.vit_name)
    
        
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/content/TransUNet/Data/Synapse/train_npz',
            'list_dir': '/content/TransUNet/lists/lists_Synapse',
            'num_classes': 9,
        },
        'Corona': {
            'root_path': '/content/TransUNet/Data/Corona19/train_npz',
            'list_dir': '/content/TransUNet/lists/lists_Corona',
            'num_classes': 2,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "/content/TransUNet/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    if args.vit_name.find('R50') != -1 or args.vit_name.find('R152') != -1:
     print(config_vit.resnet.num_layers)
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    #ResNet网络调整grid为14×14(patch_size=16)
    #‘=-1’就是指name里没有R50/R152
    if args.vit_name.find('R50') != -1 or args.vit_name.find('R152') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes,argsV=args,Flag32=flag32).cuda()
    #net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Synapse': trainer_Medicine,'Corona': trainer_Medicine}
    trainer[dataset_name](args, net, snapshot_path,Flag32=flag32)
