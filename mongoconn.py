#import lib pymongo, pip jika tidak ada
import pymongo 
#import lib time
import time
#import lib shortuuid untuk uuid random
import shortuuid
#setup connection ke mongodb
srvs = ["mongodb+srv://Admin:Author@cluster0.ruhhbnu.mongodb.net/?retryWrites=true&w=majority","mongodb://Admin:Author@ac-0wy1zny-shard-00-00.ruhhbnu.mongodb.net:27017,ac-0wy1zny-shard-00-01.ruhhbnu.mongodb.net:27017,ac-0wy1zny-shard-00-02.ruhhbnu.mongodb.net:27017/?ssl=true&replicaSet=atlas-1xfb5q-shard-0&authSource=admin&retryWrites=true&w=majority"]
client = None
for i in srvs:
    try : 
        client = pymongo.MongoClient(i, serverSelectionTimeoutMS=3000)
        print(f"Trying using {i} for connecting to MongoDB")
    except : passdb = client.test

DB = client["ANPR"]['Data']

#fungsi add data ke mongodb
def insert(fd,plat):
    #uuid sebagai identifier
    uuid = str(shortuuid.ShortUUID().random(length=22))
    json = {
        "uuid":uuid,
        #waktu penambahan data
        "date":str(time.strftime("%d")),
        "month":str(time.strftime("%b")),
        "year":str(time.strftime("%Y")),
        # url = https://loc.com + data['raw_img]
        "raw_img":f"/static/upload/{fd}/images.jpg",
        "ocr_img":f"/static/upload/{fd}/image.jpg",
        #hasil ocr
        "ocr_result":plat
    }
    #tambah 1 data
    DB.insert_one(json)
    return uuid

def jsonize(data):
    return {"uuid":data['uuid'],'date':data['date'],'month':data['month'],'year':data['year'],'raw_img':data['raw_img'],'ocr_img':data['ocr_img'],'ocr_result':data['ocr_result']}

#fungsi mencari database
def search(uuid):
    data = DB.find_one({"uuid":uuid})
    return jsonize(data)

#fungsi untuk menampilkan semua data dari database
def getall():
    a = DB.find()
    data = []
    for i in a: 
        data.append(jsonize(i))
    return data

print(getall())