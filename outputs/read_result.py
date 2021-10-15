import json

json_open = open('/home/eikai/imageRecognition/AlphaPose/outputs/alphapose-results.json', 'r')
json_load = json.load(json_open)

for v in json_load:
    print(v["image_id"],v["keypoints"],"\n")
    #print(v["image_id"],v["keypoints"][0:2],"\n")
    if int(v["image_id"][0:3])>150:
        break