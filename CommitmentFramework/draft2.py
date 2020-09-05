# 2 gray and resize

# from PIL import Image
# import os.path
# import glob
#
# cnt=1
#
# def convertjpg(jpgfile,outdir,width=35,height=35):
#     global cnt
#     print(cnt)
#     img=Image.open(jpgfile).convert('L')
#     try:
#         new_img=img.resize((width,height),Image.BILINEAR)
#         new_img.save(os.path.join(outdir,f'{cnt}.jpg'))
#         cnt+=1
#     except Exception as e:
#         print(e)
# for jpgfile in glob.glob(r"C:/Users/zhaobo/Desktop/793114_1361644_bundle_archive/CelebDataProcessed/Alyssa_Milano/*.jpg"):
#     convertjpg(jpgfile,"C:/Users/zhaobo/Desktop/1")

#2 csv
#
import csv,os,cv2
def convert_img_to_csv(img_dir):
    #设置需要保存的csv路径
    with open(r"C:/Users/zhaobo/Desktop/subject1/cm-center.csv","w",newline="") as f:
        #设置csv文件的列名
        column_name = ["label"]
        column_name.extend(["pixel%d"%i for i in range(35*35)])
        #将列名写入到csv文件中
        writer = csv.writer(f)
        writer.writerow(column_name)
        #该目录下有9个目录,目录名从0-9
        for i in range(2):
            #获取目录的路径
            img_temp_dir = os.path.join(img_dir,str(i))
            #获取该目录下所有的文件
            img_list = os.listdir(img_temp_dir)
            #遍历所有的文件名称
            for img_name in img_list:
                #判断文件是否为目录,如果为目录则不处理
                if not os.path.isdir(img_name):
                    #获取图片的路径
                    img_path = os.path.join(img_temp_dir,img_name)
                    #因为图片是黑白的，所以以灰色读取图片
                    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                    #图片标签
                    row_data = [i]
                    #获取图片的像素
                    row_data.extend(img.flatten())
                    #将图片数据写入到csv文件中
                    writer.writerow(row_data)

if __name__ == "__main__":
    #将该目录下的图片保存为csv文件
    convert_img_to_csv(r"C:/Users/zhaobo/Desktop/cm-center")

# data enhancement
#
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import load_img
# # from keras.preprocessing.image import *
#
# def date_enhancement(img_input_path,img_output_path):
#     image = load_img(img_input_path)
#     image = img_to_array(image) #图像转为数组
#     image = np.expand_dims(image, axis=0) #增加一个维度
#     img_dag = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#                             height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2,
#                             horizontal_flip = True, fill_mode = "nearest") #旋转，宽度移动范围，高度移动范围，裁剪范围，水平翻转开启，填充模式
#
#     img_generator = img_dag.flow(image, batch_size=1,
#                                  save_to_dir=img_output_path,
#                                  save_prefix = "image", save_format = "jpg")#测试一张图像bath_size=1
#     count =0 #计数器
#     for img in img_generator:
#         count += 1
#         if count == 15:  #生成多少个样本后退出
#             break
#
# if __name__=="__main__":
#     for i in range(1,78):
#         image_path =f"C:/Users/zhaobo/Desktop/1/{i}.jpg"
#         image_out_path = "C:/Users/zhaobo/Desktop/1_expanded"
#         date_enhancement(image_path,image_out_path)