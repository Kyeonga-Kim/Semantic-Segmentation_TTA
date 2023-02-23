import os
import glob
import shutil

extension = 'jpg'
file_list = os.path.join("./nas/VOCdevkit/VOC2012/ImageSets/Segmentation", "val.txt")
file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
print(len(file_list))

#image_files = sorted(os.path.join("./nas/VOCdevkit/VOC2012/JPEGImages/"))
# image_files = sorted(os.listdir('./nas/VOCdevkit/VOC2012/JPEGImages/'))
# dst = "./nas/VOCdevkit/VOC2012/JPEGImages_val"
# # print(image_files)
# for file in file_list:
#     val_file = file[0] + '.jpg'
#     if val_file in image_files:
#         ori = os.path.join('./nas/VOCdevkit/VOC2012/JPEGImages/', val_file)
#         dst_files = os.path.join(dst, val_file)    
#         shutil.copy(ori, dst_files)