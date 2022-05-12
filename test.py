import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
import copy

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
def RGB(x):
        a1 = copy.deepcopy(x) #R
        # print(f'修改前a1的shape:{a1.shape}')
        a2 = copy.deepcopy(x) #G
        a3 = copy.deepcopy(x) #B

        a1[a1 == 1] = 0
        a1[a1 == 2] = 0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 0
        a1[a1 == 5] = 255
        a1[a1 == 6] = 255
        a1[a1 == 7] = 137
        a1[a1 == 8] = 245

        a2[a2 == 1] = 0
        a2[a2 == 2] = 255
        a2[a2 == 3] = 64
        a2[a2 == 4] = 229
        a2[a2 == 5] = 0
        a2[a2 == 6] = 236
        a2[a2 == 7] = 104
        a2[a2 == 8] = 245

        a3[a3 == 1] = 128
        a3[a3 == 2] = 0
        a3[a3 == 3] = 64
        a3[a3 == 4] = 238
        a3[a3 == 5] = 255
        a3[a3 == 6] = 139
        a3[a3 == 7] = 205
        a3[a3 == 8] = 245

        a1=a1.transpose([1,2,0])
        a2=a2.transpose([1,2,0])
        a3=a3.transpose([1,2,0])

        listt=[]
        for i in range(a1.shape[2]):
         r = Image.fromarray(np.uint8(a1[:,:,i])).convert('L')
         g = Image.fromarray(np.uint8(a2[:,:,i])).convert('L')
         b = Image.fromarray(np.uint8(a3[:,:,i])).convert('L')
         x = Image.merge('RGB', [r, g, b])
         x = np.asarray(x)
         listt.append(x)
        return listt

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
#计算准确率，预测结果中占label的面积
def Right(output, target):

    if torch.is_tensor(output):
        # output = torch.sigmoid(output).data.cpu().numpy()
        output = torch.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    H,W,C=output.shape[1],output.shape[2],output.shape[0]
    print('output:{}\ntarget:{}'.format(output.shape,target.shape))
    r1=0
    r2=0
    s=0
    for c in range(C):
      for h in range(H):
        for w in range(W):
          if output[c,h,w]==target[c,h,w]:
            r1+=1
          if output[c,h,w]==target[c,h,w] and target[c,h,w]!=0:
            r2+=1
          if target[c,h,w]!=0:
            s+=1
    
    #r1/(H*W*C)为全图准确率，比较全面地反应预测结果准确性，但可能受到多余黑色背景影响拉高准确率
    #(r2/s)为label中灰白色区域准确率，可以较为准确得出预测的准确性，但可能
    #受到预测结果溢出的影响
    #原因在于TN/FN会和周边无用的黑色像素混在一起，所以用全面准确率+TP/FP灰白色区域准确率结合
    return (r1/(H*W*C)+(r2/s))/2
        

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,Flag=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                #pred中得出的贡献最大的类，之后会和label中贡献最大的类作比较，看是不是判断对了，计算dice和hd95
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                #out改下大小就是prediction
                if x != patch_size[0] or y != patch_size[1]:
                    if Flag==True:
                        pred = zoom(out, (x / 112, y / 112), order=0)
                    else:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    #看看predictions是什么
    # print('prediction内容如下：\n{}'.format(prediction))
    # print('有{}张图片'.format(len(prediction)))
    # f=open(r'/content/TransUNet/Test_outputs/predictions/predictions_p.txt','a')
    # pred_states=str(prediction)
    # f.write('prediction:{}\n维度是{}\n'.format(pred_states,str(pred.shape)))
    # f.close()
    print('\nprediction的最大值是{}，最小值是{}'.format(prediction.max(),prediction.min()))
    metric_list = []
    right=0.0
    #计算每张图8个类的Dice和HD95性能指标

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    #计算准确率
    right=Right(prediction, label)
   
     
    if test_save_path is not None:
        pred_list=RGB(prediction)
        #pred_list里每个元素大小为(512,512,3),共有原图C个元素(图1是139个)
        np.savez(test_save_path + '/'+ case + "_pred.npz", pred=pred_list)

        
        label_list=RGB(label)
        #pred_list里每个元素大小为(512,512,3),共有原图C个元素(图1是139个)
        np.savez(test_save_path + '/'+ case + "_label.npz", label=label_list)

        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # prd_itk = sitk.GetImageFromArray(pred_list.astype(np.float32))
        # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        # prd_itk.SetSpacing((1, 1, z_spacing))
        # lab_itk.SetSpacing((1, 1, z_spacing))
        # sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        # sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list,right


