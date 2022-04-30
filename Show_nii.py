encoding='utf8'
'''
查看和显示nii文件
'''

import matplotlib
import os

matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
from scipy import ndimage

# example_filename = r'D:\MedSegment\TransUnet_Chy\test_outputs'
example_filename=r'D:\MedSegment\TransUnet_Chy\test_outputs\Corona'
img_name=os.listdir(example_filename)
lenimg=len(img_name)
img=[]
for i in range(lenimg):
 imgi = nib.load(os.path.join(example_filename,img_name[i]))
 img.append(imgi.dataobj)
# vit_name_list=['Original','label','Vit-B_16','Vit-B_32','Vit-L_16','R50-Vit-B_16','R50-Vit-L_16','R152-Vit-B_16']
vit_name_list=['Original','label','R50-Vit-L_16']
'''3D图像显示'''
# OrthoSlicer3D(img.dataobj).show()

ind = 1
for i in range(lenimg):
    img_arr = img[i][:, :, 146]
    img_arr=ndimage.rotate(img_arr,120,reshape=False)  #逆时针旋转90°
    plt.subplot(1, 3, ind)  #3行3列，9张图
    plt.title(vit_name_list[i])
    plt.axis('off')
    plt.imshow(img_arr, cmap='gray')
    ind += 1

plt.show()