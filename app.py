from flask import Flask, render_template, request
import os 
from deeplearning import object_detection
from deeplearning import OCR
import shutil
import csv
import time
from mongoconn import *

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

def getfd():
    idxs = 0
    while True:
        try : 
            os.mkdir(f"./static/upload/{idxs}")
            return idxs
        except Exception as E: 
            idxs += 1

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        fd = getfd()
        path_save = os.path.join(UPLOAD_PATH+str(fd)+"/"+'images.jpg')
        upload_file.save(path_save)
        a = object_detection(path=(os.path.join(UPLOAD_PATH+str(fd)+"/"+'images.jpg')),filename='image.jpg')
        shutil.copy('./static/roi/image.jpg', os.path.join(UPLOAD_PATH+str(fd)+"/"+'image.jpg')) #memindahkan file image.jpg dari folder roi ke folder upload
        plat = OCR('./static/roi/image.jpg')
        with open("database.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            if len(plat) == 0: plat="Undetected"
            writer.writerow([fd,time.strftime("%d"),time.strftime("%b"),time.strftime("%Y"),f"/static/upload/{fd}/images.jpg",f"/static/upload/{fd}/image.jpg",plat])
        #print(str(fd)+"/images.jpg",plat,len(plat),fd)
        return render_template('index.html',upload=True,upload_image=str(fd)+"/images.jpg",text=plat,no=len(plat),idx=fd)
    return render_template('index.html',upload=False)

@app.route('/history',methods=['POST','GET'])
def history():
    rows = []
    M = None
    Y = 2023
    if request.method == 'POST':
        M = request.form['Month']
        Y = request.form['Year']
        print(M,Y)
        with open("database.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if M and Y != None:
                    if M == "all": 
                        if row[3] == Y: rows.append(row)
                    elif row[2] == M and row[3] == Y:
                        rows.append(row)
                
    else :
        with open("database.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)
    print(rows)
    return render_template('history.html',data=rows,M=M,Y=Y)

#fungsi api
#1 url untuk post dan get
@app.route('/api',methods=['POST','GET'])
def api():
    #logic jika menerima post request berbentuk file
    if request.method == 'POST':
        #logic untuk mencoba mengambil file
        try : upload_file = request.files['image']
        #jika gagal
        except : return {"Message":f"Error, Please POST requests a image"}
        #jika berhasil
        fd = getfd()
        path_save = os.path.join(UPLOAD_PATH+str(fd)+"/"+'images.jpg')
        upload_file.save(path_save)
        #plat = OCR(os.path.join(UPLOAD_PATH+str(fd)+"/"+'images.jpg'))
        a = object_detection(path=(os.path.join(UPLOAD_PATH+str(fd)+"/"+'images.jpg')),filename='image.jpg')
        shutil.copy('./static/roi/image.jpg', os.path.join(UPLOAD_PATH+str(fd)+"/"+'image.jpg'))
        plat = OCR('./static/roi/image.jpg')
        uuid = insert(fd,plat)
        return {"Message":f"Sucess, UUID : {uuid}, Please GET requests with this UUID to get the extracted data","UUID":uuid,"Data":search(uuid)}
    #logic jika menerima get request bersama uuid
    elif request.method == 'GET':
        #logic untuk mendapatkan value key UUID
        try : uuid = request.args.get('UUID')
        #jika gagal
        except : return {"Message":f"Error, Please GET requests with key 'UUID' and received UUID from POST requesting a image"}
        #jika berhasil
        data = search(uuid)
        return data
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image">
      <input type="submit">
    </form>
    '''

@app.route('/api/all',methods=['GET'])
def api_all():
    return {"data":getall()}

if __name__ =="__main__":
    app.run(debug=True)