import imgaug as ia
import cv2 as cv
import numpy as np
import os
import shutil
import re
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

aug1 = iaa.Dropout(p=0.2)
aug2 = iaa.AverageBlur(k=(5, 20))
aug3 =  iaa.Add((-40, 40), per_channel=0.5)
aug4 = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
aug5 = iaa.Affine(rotate=(-20,20))

def augment_pic( img , aug ):# 画像に変換を適用する
    aug_img = aug.augment_image(img)
 
    # 描画
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(aug_img)
    plt.show()

def augment( img , bb , aug ):
    # 画像とバウンディングボックスを変換
    aug_img = aug.augment_image( img ) 
    aug_bb = aug.augment_bounding_boxes([bb])[0].remove_out_of_image().cut_out_of_image()
    
    # バウンディングボックスと画像を重ねる
    image_before = bb.draw_on_image(img, thickness=2, color=[255, 0,0])
    image_after = aug_bb.draw_on_image(aug_img, thickness=2, color=[0, 255, 0])

    '''
    # 変換前後の画像を描画
    fig = plt.figure()
    fig.add_subplot(121).imshow(image_before)
    fig.add_subplot(122).imshow(image_after)
    plt.show()
    '''
    
    return aug_img , aug_bb

"""xmlからバウンディングボックスを取る。もしかするともっと良い関数があるかもしれない。"""
def From_xml(filename):
    tree = ET.parse( filename )
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    label = []
    boxes = []
    
    xmins = tree.findall('object/bndbox/xmin')
    for xmin in xmins:
        x1.append(xmin.text)
    xmaxs = tree.findall('object/bndbox/xmax')
    for xmax in xmaxs:
        x2.append(xmax.text)
    ymins = tree.findall('object/bndbox/ymin')
    for ymin in ymins:
        y1.append(ymin.text)
    ymaxs = tree.findall('object/bndbox/ymax')
    for ymax in ymaxs:
        y2.append(ymax.text)
    names = tree.findall('object/name')
    for name in names:
        label.append(name.text)
    width = tree.findall('size/width')[0].text
    height = tree.findall('size/height')[0].text    
        
    for i in range(len(x1)):
        boxes.append(ia.BoundingBox(x1=float(x1[i]), y1=float(y1[i]), x2=float(x2[i]), y2=float(y2[i]),label=label[i]))
    
    # バウンディングボックスを定義
    bb = ia.BoundingBoxesOnImage( boxes , shape=(int(width),int(height),3))
    
    return bb

print(From_xml("2018_test.xml"))

"""バウンディングボックスからxmlを作る。save_dirは最後に／が必要なので注意"""

#XMLファイルの生成
def To_Xml( filename , bb , save_dir = ""):
        Annotation = ET.Element('annotation')
        Filename = ET.SubElement(Annotation,'filename')
        Filename.text = filename +'.jpg'

        size = ET.SubElement(Annotation,'size')
        width = ET.SubElement(size,'width')
        width.text = str(bb.shape[0])
        height = ET.SubElement(size,'height')
        height.text = str(bb.shape[1])

        for i in range(len(bb.bounding_boxes)):
            Object = ET.SubElement(Annotation, 'object')
            name = ET.SubElement(Object, 'name')
            name.text = str(bb.bounding_boxes[i].label)
            bndbox = ET.SubElement(Object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(bb.bounding_boxes[i].x1)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(bb.bounding_boxes[i].y1)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(bb.bounding_boxes[i].x2)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(bb.bounding_boxes[i].y2)
    
        tree = ET.ElementTree(element=Annotation)
        
        #保存
        filename = save_dir + filename + '.xml'
        tree.write(filename, encoding='utf-8', xml_declaration=True)

"""ファイルを連番にする"""

def renban(file_path , save_path):
    files = os.listdir(path = file_path)
    print(len(files))
    for i in range(int(len(files)/2)):
        print(files[2*i])

        shutil.copyfile(file_path+'/'+files[2*i] ,save_path+'/'+str(i)+'.jpg')
        bb = From_xml(file_path+'/'+files[2*i+1])
        To_Xml( str(i) , bb , save_dir = save_path+"/")

"""アノテーションデータを水増しする"""

#XMLファイルの生成
def To_Resized_Xml( filename , bb , pic_size, save_dir = ""):
        Annotation = ET.Element('annotation')
        Filename = ET.SubElement(Annotation,'filename')
        Filename.text = filename +'.jpg'

        size = ET.SubElement(Annotation,'size')
        width = ET.SubElement(size,'width')
        width.text = str(pic_size)
        height = ET.SubElement(size,'height')
        height.text = str(pic_size)

        bai = pic_size/bb.shape[0]
        
        for i in range(len(bb.bounding_boxes)):
            Object = ET.SubElement(Annotation, 'object')
            name = ET.SubElement(Object, 'name')
            name.text = str(bb.bounding_boxes[i].label)
            bndbox = ET.SubElement(Object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(int(bai* bb.bounding_boxes[i].x1))
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(int(bai* bb.bounding_boxes[i].y1))
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(int(bai* bb.bounding_boxes[i].x2))
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(int(bai* bb.bounding_boxes[i].y2))
    
        tree = ET.ElementTree(element=Annotation)
        
        #保存
        filename = save_dir + filename + '.xml'
        tree.write(filename, encoding='utf-8', xml_declaration=True)
        
def Resize( size ,img ):
    #サイズは横のほうが大きいと決めつけ
    height, width = img.shape[:2]
    bai = size/width
    h = int(height*bai)
    w = size
    img2 = cv.resize(img,( w,h ))
    img_re = np.ones(( size, size, 3),np.uint8)
    img_re[0:h, 0:w] = img2
    
    return img_re

def main(file_path , save_path , size , mode= [aug1,aug2,aug3,aug4,aug5]):
    files = os.listdir(path = file_path)
    num = int(len(files)/2)
    print("file num is " + str(num))

    xmlfiles = []
    regex = re.compile(r'(.xml)$')
    for name in files:   #filesは上記例で得られたリスト
      if regex.search(name):
        xmlfiles.append(name)

    for file in xmlfiles:
      img = cv.imread(file_path + '/' + file.split('.')[0] + '.jpg')
      bb = From_xml(file_path + '/' + file)

      for j in range(len(modes)):
          aug_img , aug_bb = augment(img , bb ,modes[j])  
          resized_img = Resize( size , aug_img )
            
          cv.imwrite(save_path + '/' + file.split('.')[0] + '_' + str(j) +".jpg", resized_img)
          To_Resized_Xml( file.split('.')[0] + '_' + str(j) , aug_bb , size ,save_dir = save_path+'/')

def test(file_path , save_path , size):
    files = os.listdir(path = file_path)
    num = int(len(files)/2)
    print("file num is" + str(num))

    for i in range(num):
        print(file_path + files[2*i])
    
        img = cv.imread(file_path + files[2*i])
        bb = From_xml(file_path + files[2*i+1])
        
        resized_img = Resize( size , img )
        
        cv.imwrite(save_path +str(i)+".jpg", resized_img)
        To_Resized_Xml( str(i) , bb , size ,save_dir = save_path)
  