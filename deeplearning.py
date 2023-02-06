import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image as Img
from PIL import ImageTk
# dependencies
from IPython.display import Image

import argparse
import sys
import os.path

import matplotlib.gridspec as gridspec
from os.path import splitext,basename
from keras.models import model_from_json
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob

# LOAD YOLO MODEL
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #set model ke opencv
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# LOAD CHARACTER RECOGNITION MODEL
json_file = open('./model5/MobileNets_character_recognition_akudataset.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("./model5/models_cnn.h5")

labels = LabelEncoder()
labels.classes_ = np.load('./model5/license_character_classes_akudataset.npy')


def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)  # construct a blob from the image
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4] #itu berarti row ke 0 sampai batas 4

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()
    
    return boxes_np, confidences_np, index

def extract_text(image,bbox): #untuk mengambil crop plat nomor
    x,y,w,h = bbox 
    
    roi = image[y:y+h, x:x+w]
    if 0 in roi.shape:
        return ''
    else:
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        print("img")
        cv2.imwrite('./static/roi/image.jpg',roi_bgr) #menyimpan hasil crop di folder roi

def drawings(image,boxes_np,confidences_np,index):
    # drawings hasil bounding box
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100) #menampilkan confidence dalam bentuk persen
        license_text = extract_text(image,boxes_np[ind]) #crop plat nomer


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2) #gambar bounding boxnya
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1) #gambar untuk background ungu persen

        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1) #menampilkan persen confidenc

    return image


# predictions
def yolo_predictions(img,net):
    ## step-1: detections
    input_image, detections = get_detections(img,net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img


def object_detection(path,filename):
    # read image
    image = cv2.imread(path) # PIL object
    result_img = yolo_predictions(image,net)
    cv2.imwrite('./static/predict/{}'.format(filename),result_img) #menyimpan hasil prediciton ke folder predict
    return result_img

def find_contours(dimensions, img, img_path) : # menerim tigaa argumen
    print(img_path)
    ii = cv2.imread(img_path)

    #menemukan kontur
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0] # batas bawah
    upper_width = dimensions[1] # batas atas dari lebar
    lower_height = dimensions[2] # batas bawah dari tinggi
    upper_height = dimensions[3] #batas atas dari tinggi
    
    # memeriksa kontur yang memungkinkan dibates ada 12
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:12]
    
    x_cntr_list = []
    img_res = []
    for cntr in cntrs :
        # variabel integer nyimpen koordinat x dan y, lebar dan tinggi
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # digunakan untuk memfilter kontur yang memenuhi syarat ukuran lebar dan tinggi yang diinginkan. 
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            # membuat suatu array baru dengan ukuran 44x24
            char_copy = np.zeros((44,24))
            # Array numpy char diinisialisasi dengan bagian dari img yang memiliki koordinat (intY, intX) dan ukuran (intHeight, intWidth)
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            cv2.drawContours(ii, [cntr], 0, (0, 255, 0), 5)
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (255,0,0), 5)

            char = cv2.subtract(255, char)

            # memasukkan citra "char" yang diambil sebelumnya dan telah diresize ke dalam array "char_copy"
            char_copy[2:42, 2:22] = char
            # mengatur sekeliling citra dengan mengisi sekeliling citra dengan nilai 0. 
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) 

    # Mengurutkan list img_res berdasarkan urutan x_cntr_list           
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx]) #menyimpan gambar karakter sesuai dengan indeksnya
    img_res = np.array(img_res_copy) # mengubah menjadi array numpy

    # sebuah array yang berisi gambar karakter yang sudah disort berdasarkan urutan posisi x setiap kontur karakter
    return img_res

def segment_characters(image) :
    # Preprocess cropped license plate image
    img_lp = cv2.imread(image)
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    # menentukan lebar dan tinggi 
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Estimasi ukuran kontur karakter pelat nomor
    dimensions = [LP_WIDTH/8,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/8]

    # manggil fungsi find_countours
    char_list = find_contours(dimensions, img_binary_lp,image)
    #  Mengembalikan list "char_list" yang berisi informasi kontur karakter dalam plat nomor.
    return char_list

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    # melakukan prediksi dari model dengan mengambil index dari elemen maksimal dari hasil prediksi, lalu mengubah index jadi label
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def OCR(img_path):
    digits = segment_characters(img_path)
    final_string = ''
    for i,character in enumerate(digits):
        # memanggil fungsi prediksi dan mengubah hasil prediksi jadi string
        title = np.array2string(predict_from_model(character,model,labels))
        # menambahkan hasil prediksi karakter pada string akhir, outputnya itu "['A]" makanya pake .strip biar jadi A
        final_string+=title.strip("'[]")
    return final_string

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf
