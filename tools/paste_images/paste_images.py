#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image

COL = 1 #指定拼接图片的列数
ROW = 2 #指定拼接图片的行数
UNIT_HEIGHT_SIZE = 768 #图片高度
UNIT_WIDTH_SIZE = 1024 #图片宽度
SAVE_QUALITY = 100 #保存的图片的质量 可选0-100

# path=input('请输入文件路径(结尾加上/)：')       
path_pro='./pro'
path_fow='./fow'
path_save = './combinedImages'
#获取该目录下所有文件，存入列表中
fileList_pro=os.listdir(path_pro)
fileList_fow=os.listdir(path_fow)

n = 1
for pro_fileName,fow_fileName in zip(fileList_pro,fileList_fow):
    pro_image = Image.open(path_pro + '/' + pro_fileName)
    fow_image = Image.open(path_fow + '/' + fow_fileName)
    
    # 创建成品图的画布
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))
    
    # paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
    target.paste(pro_image, (UNIT_WIDTH_SIZE*0, UNIT_HEIGHT_SIZE*0))
    target.paste(fow_image, (UNIT_WIDTH_SIZE*0, UNIT_HEIGHT_SIZE*1))
    
    # Name and save image
    nString = str(n)
    target.save(path_save +'/' + nString + '.jpg', quality=SAVE_QUALITY)
    n+=1

print('jobs done!')


# In[ ]:




