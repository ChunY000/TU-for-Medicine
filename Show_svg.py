
import cairosvg
import os
from PIL import  Image
import matplotlib.pyplot as plt

'''svg2png'''
# svg_listdir='D:\MedSegment\TransUnet_Chy\loss_record_svg'
# png_listdir='D:\MedSegment\TransUnet_Chy\loss_record_png'
#
# svg_name=os.listdir(svg_listdir)
# for i in range(len(svg_name)):
#     svgfile_path=os.path.join(svg_listdir,svg_name[i])
#     png_name=str(svg_name[i])+'.png'
#     png_savedir=os.path.join(png_listdir,png_name)
#     png_savedir = png_savedir.encode('gbk')
#     cairosvg.svg2png(url=svgfile_path, write_to=png_savedir)

'''show png'''
png_listdir='D:\MedSegment\TransUnet_Chy\loss_record_png'
file_name=os.listdir(png_listdir)
ind=1
for i in range(len(file_name)):
    img=Image.open(os.path.join(png_listdir,file_name[i]))
    plt.subplot(3, 4, ind)
    plt.title(file_name[i][:-4])
    plt.axis('off')
    plt.imshow(img)
    ind+=1

plt.show()