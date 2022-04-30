import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_Medicine import Medicine_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/content/TransUNet/Data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='/content/TransUNet/lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=False, help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--GoogleUse', type=bool,default=False, help='Whether or not using the pretrained Models of google')
parser.add_argument('--Flag32', type=bool,default=False, help='Whether or not using the 32 VIT')
args = parser.parse_args()


def inference(args, model, test_save_path=None,Flag32=False):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    right_list=[]
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        metric_i,right = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,Flag=Flag32)
        
        metric_list += np.array(metric_i)
        right_list.append(right)
        #np.mean就是对这张图的8个类求均值，所以也是这张图的均值
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch+1, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        if args.GoogleUse==False:
          vit_name='OwnTraining_{}_{}'.format(args.datasets,args.vit_name)
        else:
          vit_name=args.vit_name
        DinaryLoss_path='/content/gdrive/MyDrive/TransUnet_Chy/DinaryLoss/{}.txt'.format(vit_name)
        F=open(DinaryLoss_path,'a')
        if i_batch==0:
         F.write('误差数据如下(左边Dice，右边hd95,下面准确率):\n')
        F.write(str('图%d的误差： %s mean_dice %f mean_hd95 %f\n' % (i_batch+1, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])))
        F.write(str('图%d的准确率： %.2f' % (i_batch+1, right*100)+'%\n'))
        logging.info(str('图%d的准确率： %.2f' % (i_batch+1, right*100))+'%\n')
    idxDice_Sum = 0
    idxHd95_Sum = 0
    right_Sum=0
    lenidx=len(metric_list)

    #metric_list里记录的是8个类别的dice和hd95，最后/8得到的是所有图的总DICE和总HD95，还要/4
    for ii in range(0,lenidx):
      idxDice_Sum += metric_list[ii][0]
      idxHd95_Sum += metric_list[ii][1]
    idxDice_Mean=float(idxDice_Sum/(lenidx*len(testloader)))
    idxHd95_Mean=float(idxHd95_Sum/(lenidx*len(testloader)))
    
    for n in range(len(testloader)):
      right_Sum+=right_list[n]
    right_Mean=right_Sum/len(testloader)
    
    print('lenidx={},idxDice_Sum={},idxDice_Mean={}'.format(lenidx,idxDice_Sum,idxDice_Mean))
    F.write(str('所有图的均误差： mean_dice %f mean_hd95 %f\n' % (idxDice_Mean, idxHd95_Mean)))
    F.write(str('所有图的均准确度： %.2f' % (right_Mean*100))+'%\n')
        
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        F.write(str('第%d类均误差：mean_dice %f mean_hd95 %f\n' % (i, metric_list[i-1][0], metric_list[i-1][1])))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    F.write(str('总类均误差: mean_dice : %f mean_hd95 : %f\n' % (performance, mean_hd95)))
    F.close()
    return "测试完成!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Medicine_dataset,
            'volume_path': '/content/TransUNet/Data/Synapse/test_vol_h5',
            'list_dir': '/content/TransUNet/lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'Corona': {
            'Dataset': Medicine_dataset,
            'volume_path': '/content/TransUNet/Data/Corona19/test_vol_h5',
            'list_dir': '/content/TransUNet/lists/lists_Corona',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    flag32=False
    data_choose=int(input('你想要测试什么1.Synapse 2.Corona:'))
    if data_choose ==1:
      args.dataset='Synapse'
    elif data_choose ==2:
      args.dataset='Corona'
    UseGoogle=input('要使用谷歌预训练模块吗？ 1.使用 2.不使用')
    UseG=int(UseGoogle)
    if UseG==1:
      args.GoogleUse=True 
      ModelType=input('1.VIT-B-16  2.VIT-B-32  3.VIT-L-16  4.R50-VIT-B-16  5.R152-VIT-B-16  6.R50-VIT-L-16:')
      IntMT=int(ModelType)
      if IntMT == 1:
        args.vit_name='ViT-B_16'
        args.vit_patches_size=16
      elif IntMT==2:
        args.vit_name='ViT-B_32'
        args.vit_patches_size=32
        args.Flag32=True
        flag32=True
      elif IntMT==3:
        args.vit_name='ViT-L_16'
        args.vit_patches_size=16
      elif IntMT==4:
        args.vit_name='R50-ViT-B_16'
        args.vit_patches_size=16
    else:
      UseOwn=int(input('想用哪个自己训练的模型 1.R50-vit-b_16  2.vit-b_16  3.vit-b_32  4.vit-l_16 5.R152 6.R50-l: '))
      if UseOwn == 2:
        args.vit_name='ViT-B_16'
      elif UseOwn == 3:
        args.vit_name='ViT-B_32'
        args.vit_patches_size=32
        args.Flag32=True
      elif UseOwn == 4:
        args.vit_name='ViT-L_16'
        args.batch_size=20
      elif UseOwn == 5:
        args.vit_name='R152-ViT-B_16'
        args.max_epochs=80
        args.n_skip=3
      elif UseOwn == 6:
        args.vit_name='R50-ViT-L_16'
        args.n_skip=3
        if args.dataset=='Corona':
          args.batch_size=20
          args.max_epochs=100
        else:
          args.batch_size=18
      else :
        args.vit_name='R50-ViT-B_16'
        args.n_skip=3
      
    Saveimg=input('要保存图片吗？ 1.保存 2.不保存') 
    SaveImg=int(Saveimg)
    if SaveImg==1:
      args.is_savenii=True
      
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    
    if args.GoogleUse==False:
    # name the same snapshot defined in train script!
       snapshot_path = "/content/gdrive/MyDrive/TransUnet_Chy/model/{}/{}".format(args.exp, 'TU')
       snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
       snapshot_path += '_' + args.vit_name
       snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
       snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
       snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
       if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
           snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
       snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
       snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
       snapshot_path = snapshot_path + '_'+str(args.img_size)
       snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
   

    if args.GoogleUse==False:
       if args.vit_name.find('R50') !=-1 or args.vit_name.find('R152') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
       net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes, argsV=args,Flag32=flag32).cuda()
       snapshot = os.path.join(snapshot_path, 'best_model.pth')
       if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
       snapshot_name = snapshot_path.split('/')[-1]
       net.load_state_dict(torch.load(snapshot))
    else:
       net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes, argsV=args).cuda()
       net.load_from(weights=np.load(config_vit.pretrained_path))
       snapshot_name = args.vit_name

    log_folder = '/content/gdrive/MyDrive/TransUnet_Chy/test_outputs/test_log' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '/content/gdrive/MyDrive/TransUnet_Chy/test_outputs/predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path,args.Flag32)
