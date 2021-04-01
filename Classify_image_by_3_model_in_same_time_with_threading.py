import h5py.defs
import h5py.utils
import h5py.h5ac
import h5py._proxy
import time
import os
from os import path
import threading
import tensorflow as tf 
import keras_preprocessing
from keras_preprocessing import image
import numpy as np
import gc
import PIL as pl

initial_time = time.time()

Using_txt_file = ['C:/Users/Public/Drug dispenser/current_drug/ready.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/stop.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_0_array.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_0.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_1_array.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_1.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_2_array.txt'
                , 'C:/Users/Public/Drug dispenser/current_drug/result_2.txt']

model_path_set = ['C:/Users/Public/Drug dispenser/current_drug/drug_model_0.h5'
                , 'C:/Users/Public/Drug dispenser/current_drug/drug_model_1.h5'
                , 'C:/Users/Public/Drug dispenser/current_drug/drug_model_2.h5']

data_0_path = ['C:/Users/Public/Drug dispenser/current_drug/A.png', 
                'C:/Users/Public/Drug dispenser/current_drug/result_0_array.txt', 
                'C:/Users/Public/Drug dispenser/current_drug/result_0.txt']
data_1_path = ['C:/Users/Public/Drug dispenser/current_drug/B.png',
                'C:/Users/Public/Drug dispenser/current_drug/result_1_array.txt',   
                'C:/Users/Public/Drug dispenser/current_drug/result_1.txt']
data_2_path = ['C:/Users/Public/Drug dispenser/current_drug/C.png',
                'C:/Users/Public/Drug dispenser/current_drug/result_2_array.txt', 
                'C:/Users/Public/Drug dispenser/current_drug/result_2.txt']

initial_pic_path = 'C:/Users/Public/Drug dispenser/current_drug/initial_pic.png' 

img_0_size = 299, 299
img_1_size = 224, 224
img_2_size = 224, 224

img_size = [img_0_size, img_1_size, img_2_size]
data_path = [data_0_path, data_1_path, data_2_path]

class ManageFile():
    def __init__(self, file_path, manage_command):
        self.file_path = file_path
        self.manage_command = manage_command
        ManageFile.CheckFile(self)

    def CheckFile(self):
        check_file = os.path.exists(self.file_path)
        if self.manage_command >= 0 and self.manage_command <= 2:
            if self.manage_command == 0:
                return check_file 
            if check_file:
                if self.manage_command == 2:
                    ManageFile.RemoveFile(self)
            else:
                if self.manage_command == 1:
                    ManageFile.CreateFile(self)   
        else:
            return 'Error command'
            
    def CreateFile(self):
        create_file = open(self.file_path, 'a')
        create_file.close()
        #print('Create file done.')

    def RemoveFile(self):
        os.remove(self.file_path)

def InitialTxt(Using_txt_file):
    for i in range (len(Using_txt_file)):
        if i <= 1:
            ManageFile(Using_txt_file[i], 3)
        elif i > 1:
            ManageFile(Using_txt_file[i], 1)


def ImportModel(ModelPath):
    model = tf.keras.models.load_model(ModelPath)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    return model

def ImgToArray(UsingModel,PicPath,ImgSize,ArrayPath,ResultPath,InitialThread):
    drug_img = tf.keras.preprocessing.image.load_img(PicPath, grayscale=False, color_mode='rgb', target_size=ImgSize,interpolation='bicubic')
    data_array = tf.keras.preprocessing.image.img_to_array(drug_img, dtype = 'float32')
    data_array = np.divide(data_array, 255.)
    data_array = np.expand_dims(data_array, axis=0)
    classes = UsingModel.predict(data_array)
    classes_array = classes[0]

    maxValue = np.amax(classes_array)
    result_class = np.where(classes_array == maxValue)
    get_class = result_class[0]
    predicted_result = get_class[0]

    if InitialThread == False:
        result_array = open(ArrayPath,'w')
        result_array.write(str(classes_array))
        result_array.close()

        result = open(ResultPath,'w')
        result.write(str(predicted_result))
        result.close()

        os.remove(PicPath)

    print(predicted_result)

def RescaleArray(DataArray):
    data_array = DataArray
    data_array = np.divide(data_array, 255.)
    data_array = np.expand_dims(data_array, axis=0)

def InitialPic():
    if os.path.exists(initial_pic_path) == False:
        img_w, img_h = 299, 299
        data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        data[100, 100] = [255, 0, 0]
        img =  pl.Image.fromarray(data, 'RGB')
        img.save(initial_pic_path)

InitialTxt(Using_txt_file)
InitialPic()

model_0 = ImportModel(model_path_set[0])
model_1 = ImportModel(model_path_set[1])
model_2 = ImportModel(model_path_set[2])
print('Model 0 is ',model_0)
print('Model 1 is ',model_1)
print('Model 2 is ',model_2)

p0 = threading.Thread(target=ImgToArray, args=(model_0,initial_pic_path,img_size[0],data_0_path[1],data_0_path[2],True,))
p1 = threading.Thread(target=ImgToArray, args=(model_1,initial_pic_path,img_size[1],data_0_path[1],data_0_path[2],True,))
p2 = threading.Thread(target=ImgToArray, args=(model_2,initial_pic_path,img_size[2],data_0_path[1],data_0_path[2],True,))

p0.start()
p1.start()
p2.start()

p0.join()
p1.join()
p2.join()

print('Boosting duration is ', time.time()-initial_time,'sec')
ready_file = open(Using_txt_file[0], 'a')
ready_file.close()
print('Start classify...')

while True:
        
    gc.collect()

    pic_0_status = os.path.exists(data_0_path[0])
    pic_1_status = os.path.exists(data_1_path[0])
    pic_2_status = os.path.exists(data_2_path[0])

    if pic_0_status and pic_1_status and pic_2_status:
        begin_timer = time.time()
        p0 = threading.Thread(target=ImgToArray, args=(model_0,data_0_path[0],img_size[0],data_0_path[1],data_0_path[2],False,))
        p1 = threading.Thread(target=ImgToArray, args=(model_1,data_1_path[0],img_size[1],data_1_path[1],data_1_path[2],False,))
        p2 = threading.Thread(target=ImgToArray, args=(model_2,data_2_path[0],img_size[2],data_2_path[1],data_2_path[2],False,))

        p0.start()
        p1.start()
        p2.start()

        p0.join()
        p1.join()
        p2.join()

        pic_0_status = os.path.exists(data_0_path[0])
        pic_1_status = os.path.exists(data_1_path[0])
        pic_2_status = os.path.exists(data_2_path[0])

        while pic_0_status or pic_1_status or pic_2_status:
            pic_0_status = os.path.exists(data_0_path[0])
            pic_1_status = os.path.exists(data_1_path[0])
            pic_2_status = os.path.exists(data_2_path[0])

            if pic_0_status:
                p00 = threading.Thread(target=ImgToArray, args=(model_0,data_0_path[0],img_size[0],data_0_path[1],data_0_path[2],False,))
                p00.start()
            if pic_1_status:
                p11 = threading.Thread(target=ImgToArray, args=(model_1,data_1_path[0],img_size[1],data_1_path[1],data_1_path[2],False,))
                p11.start()
            if pic_2_status:
                p22 = threading.Thread(target=ImgToArray, args=(model_2,data_2_path[0],img_size[2],data_2_path[1],data_2_path[2],False,))
                p22.start()

            if pic_0_status:
                p00.join()
            if pic_1_status:
                p11.join()
            if pic_2_status:
                p22.join()

            time.sleep(0.05)

        print('Predicted duration is ',time.time()-begin_timer,'sec.')
        time.sleep(0.05)

