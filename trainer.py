import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from torch.autograd import Variable

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def trainer_Medicine(args, model, snapshot_path,Flag32=False):
    from datasets.dataset_Medicine import Medicine_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Medicine_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    if args.dataset == 'Corona':
        num_workers=0
    else:
        num_workers=8
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
#             if Flag32:           
#               outputs=nn.functional.interpolate(outputs,224)  
            ''''''
            # print('\noutputs:{}\nlabel_batch:{}'.format(outputs.shape,label_batch.shape))
            # if Flag32:
            #   outputs_m=outputs.cpu().detach().numpy()
            #   outputs=outputs.cpu().detach().numpy()

            #   outputs_m=outputs_m.reshape(24,1,224,504)
            #   outputs_m=outputs_m.squeeze()

            #   outputs_m=zoom(outputs_m,(1,1,224/504))
            #   outputs=zoom(outputs,(1,1,2,2))

            #   outputs_m=torch.tensor(outputs_m)
            #   outputs=torch.tensor(outputs)

            #   outputs_m=outputs_m.cuda()
            #   outputs=outputs.cuda()
            #outputs_m(24,224,224),outputs(24,9,224,224),label(24,224,224)
            # print('\noutputs:{}\nlabel_batch:{}'.format(outputs.shape,label_batch[:].long().shape))
            # if Flag32:
            #  loss_ce = ce_loss(outputs, label_batch.long())
            #  #原来是loss_ce = ce_loss(outputs_m, label_batch[:].long())
            # else:
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            # if Flag32:
            #  loss=Variable(loss,requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('循环第 %d 次 : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break

    writer.close()
    return "训练完毕!"
