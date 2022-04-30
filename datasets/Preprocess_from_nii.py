import argparse
from pathlib import Path
from typing import List
from os import listdir as ld
from os.path import join as pj

import h5py
import nibabel
import numpy
from tqdm import tqdm

#要是有现成list的记事本，记录了所有文件名
def get_case_ids_from_list(dataset_list_path: Path) -> List[str]:
    with open(dataset_list_path, "r") as f:
        slices = f.readlines()
    ##获取id这里需要按照自己的名字改下，sorted返回一个列表
    case_ids = sorted(list(set([s.split("_")[0][4:].rstrip() for s in slices])))
    return case_ids

#要没有list(我们一般选这个，list由listMaker制作)
def get_case_ids_from_directory(directory: Path) -> List[str]:
    return [f.stem for f in directory.iterdir()]


def main(args: argparse.Namespace):
    image_dir = args.original_dataset_dir / "img"  ##dir/img的操作，类似path.join
    """提取文件名.nii的列表"""
    print(image_dir)
    if args.from_list_file is not None:
        case_ids = get_case_ids_from_list(args.from_list_file)
    else:
        case_ids = get_case_ids_from_directory(image_dir)

    case_id_new=[]
    num = 10000
    for i in range(len(case_ids)):
        id=str(num+i+1)[1:]
        case_id_new.append(id)
        num=10000
    print('case_id修改版:{}'.format(case_id_new))
    print(f"case ids内容: {case_ids}")

    for case_id in tqdm(case_ids):
        case_image_dir = image_dir
        if not case_image_dir.exists():
            print(f"Sub-directory {case_image_dir} 不存在.")
            continue

        #image_path就是将根目录img+所有文件名，依次迭代形成img/xxx.nii
        for image_path in tqdm(case_image_dir.iterdir(), desc="处理中，包括裁切，归一", leave=False):
            # print(f'image_path是{image_path}')
            label_id = str(image_path.name).replace('_org','')
            # print(f'label是{label_id}')
            label_path=args.original_dataset_dir / "label" /f'{label_id}'
            # print(f'lab_path是{label_path}')


            assert image_path.exists() and label_path.exists(), '{} 没有原图，也没有label'.format(case_id)
            #载入
            image_data = nibabel.load(image_path).get_fdata()
            label_data = nibabel.load(label_path).get_fdata()
            #裁切，归一
            clipped_image_data = numpy.clip(image_data, *args.clip)
            normalised_image_data = (clipped_image_data - args.clip[0]) / (args.clip[1] - args.clip[0])  #shape(h,w,c)

            #reshape成(c,h,w)，为了让后续enumerate以c来循环，每次循环就是一次切片
            transposed_image_data = numpy.transpose(normalised_image_data, (2, 0, 1))
            transposed_label_data = numpy.transpose(label_data, (2, 0, 1))

            #当前图片的序列号
            idx = case_ids.index(str(image_path.name)[:-4])
            print(f'imgshape:{transposed_image_data.shape},labshape{transposed_label_data.shape}')
            # 分离slice
            for i, (image_slice, label_slice) in tqdm(enumerate(zip(transposed_image_data, transposed_label_data)),
                                                      desc="3D切片保存处理中", leave=False):

             out_filename = args.target_dataset_dir / args.train_save_dir /f'case{case_id_new[idx]}_slice{i:03d}.npz'
             # print(f'img_size为{image_slice.shape},lab_size为{label_slice.shape}')
             #将前80%张作为训练集，后20%作为测试集
             if idx<=int(len(case_ids)*0.8)-1:
                if not args.overwrite and out_filename.exists():  # Do not overwrite data unless flag is set
                    continue
                if not out_filename.parent.exists():
                    out_filename.parent.mkdir(exist_ok=True, parents=True)
                #存储为.npz

                numpy.savez(out_filename, image=image_slice, label=label_slice)
             else:
            # keep the 3D volume in h5 format for testing cases.
              h5_filename = args.target_dataset_dir / args.test_save_dir/f'case{case_id_new[idx]}.npy.h5'
              # print(f'orgimg—size{normalised_image_data.shape},torgimg-size{transposed_image_data.shape}')
              if not args.overwrite and h5_filename.exists():  # Do not overwrite data unless flag is set
                continue
              if not h5_filename.parent.exists():
                h5_filename.parent.mkdir(exist_ok=True, parents=True)
              with h5py.File(h5_filename, "w") as f:

                f.create_dataset("image", data=transposed_image_data)
                f.create_dataset("label", data=transposed_label_data)

def listMaker(List_path=None,get_name_path=None):
    train_name_list = ld(pj(get_name_path,'train_npz'))
    test_name_list = ld(pj(get_name_path,'test_vol_h5'))
    Ftrain = open(pj(List_path,'train.txt'), 'a')
    Ftest=open(pj(List_path,'test_vol.txt'),'a')
    for i in (train_name_list):
     iname=i[:-4]
     Ftrain.writelines(iname+'\n')
    for j in (test_name_list):
     jname = j[:-7]
     Ftest.writelines(jname+'\n')
    print('两个list文件已经保存完毕啦')


if __name__ == "__main__":
    """新数据集要改前五个地址，以及最后的lisp和get_namep"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_save_dir",type=Path,default=Path('Corona19/train_npz'))
    parser.add_argument("--test_save_dir",type=Path,default=Path('Corona19/test_vol_h5'))
    parser.add_argument("--list_save_dir", type=Path, default=Path('D:/MedSegment/TransUnet_Chy/Project2/lists/lists_Corona'))

    parser.add_argument("--original_dataset_dir", type=Path,default=Path('D:/MedSegment/TransUnet_Chy/data/Lung_corona_raw'),
                        help="The root directory for the downloaded, original dataset")

    parser.add_argument("--target-dataset-dir", type=Path, default=Path('D:/MedSegment/TransUnet_Chy/data'),
                        help="The directory where the processed dataset should be stored.")

    parser.add_argument("--from-list-file", type=Path,default=None,
                        help="Do not process all directories that are contained in the original dataset directory, "
                             "but use those contained in the passed list file. The data in the list must be "
                             "structured as in the train.txt file located in lists/lists_Synapse.")
    parser.add_argument("--clip", nargs=2, type=float, default=[-125, 275],
                        help="Two numbers [min max] that represent the interval that should be clipped from the "
                             "original image data.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite the data present in the target dataset directory")
    parsed_args = parser.parse_args()

    # 自设
    dataset_config = {
        'Corona19': {
            'root_path': 'D:/MedSegment/TransUnet_Chy/data/Lung_corona_raw',
            'train_dir': 'Corona19/train_npz',
            'test_dir': 'Corona19/test_vol_h5',
            'target_dir': 'D:/MedSegment/TransUnet_Chy/data',
            'list_dir': 'D:/MedSegment/TransUnet_Chy/Project2/lists/lists_Corona', },
        'xxx': {
            'root_path': '',
            'list_dir': '',
            'train_dir': 'xxx/train_npz',
            'test_dir': 'xxx/test_vol_h5',
            'target_dir': 'D:/MedSegment/TransUnet_Chy/data',

        }

    }
    # choose = int(input('1.corona 2.xxx'))
    # c_name = None
    # if choose == 1:
    #     c_name = 'Corona19'
    # if choose == 2:
    #     c_name = ''

    # parsed_args.original_dataset_dir = dataset_config[c_name]['root_path']
    # parsed_args.target_dataset_dir = dataset_config[c_name]['target_dir']
    # parsed_args.train_save_dir = dataset_config[c_name]['train_dir']
    # parsed_args.test_save_dir = dataset_config[c_name]['test_dir']
    # parsed_args.list_save_dir = dataset_config[c_name]['list_dir']


    listp='D:\MedSegment\TransUnet_Chy\Project2\lists\lists_Corona'
    get_namep='D:\MedSegment\TransUnet_Chy\data\Corona19'
    main(parsed_args)
    listMaker(listp,get_namep)
