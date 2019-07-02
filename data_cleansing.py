import pandas as pd
import numpy as np
import json
import os
from chainercv.visualizations import vis_bbox

annotations = os.listdir('train_annotations')

tags = []
bbox = []
i = 0
for annotation in annotations:
    with open(os.path.join('train_annotations', annotation)) as f:
        train_annotations = json.load(f)
    annotation = annotation.replace('.json', '')

    for region in train_annotations['regions']:
        new_tag = region['tags']
        new_bbox = region['boundingBox']
        tags.append(new_tag)
        bbox.append(new_bbox)

tags = np.array(tags)
bbox = np.array(bbox)



##学習データを確認する
from cv2 import cv2
import matplotlib.pyplot as plt
view_file = '05'
a = []
img = cv2.imread('train_images/train' + view_file + '.jpg')

with open('train_annotations/train' + view_file + '.json') as f:
    train_annotations = json.load(f)
    
for i in range(len(train_annotations)):
    height = train_annotations['regions'][i]['boundingBox']['height']
    width = train_annotations['regions'][i]['boundingBox']['width']
    left = train_annotations['regions'][i]['boundingBox']['left']
    top = train_annotations['regions'][i]['boundingBox']['top']
    tag = train_annotations['regions'][i]['tags'][0]
    cv2.rectangle(img, (int(left), int(top)), (int(left + width), int(top + height)), (0, 0, 255))
    cv2.putText(img, tag, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('color', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






os.makedirs('train_images_0', exist_ok=True)
os.makedirs('train_annotations_0', exist_ok=True)
os.makedirs('val_images', exist_ok=True)
os.makedirs('val_annotations', exist_ok=True)

import shutil
count = 0
for annotation in annotations:
    file_name = annotation.replace('.json', '.jpg')
    if count < 16:
        shutil.copy('train_images/' + file_name, 'train_images_0')
    else:
        shutil.copy('train_images/' + file_name, 'val_images')
    count+=1

new_annotations = {}
array_annotations = []
count = 0
for annotation in annotations:
    name = annotation.replace('.json', '')
    with open(os.path.join('train_annotations', annotation)) as f:
        train_annotations = json.load(f)
        size = len(train_annotations)
        for i in range(size):
            height = train_annotations['regions'][i]['boundingBox']['height']
            width = train_annotations['regions'][i]['boundingBox']['width']
            left = train_annotations['regions'][i]['boundingBox']['left']
            top = train_annotations['regions'][i]['boundingBox']['top']
            tag = train_annotations['regions'][i]['tags'][0]
            new_annotations[name] = {'tag':tag, 'height':height, 'width':width, 'left':left, 'top':top}
            array_annotations.append(new_annotations[name])



        if count < 16:
            with open('train_annotations_0/' + annotation, 'w') as f:
                json.dump(array_annotations, f)
        else:
            with open('val_annotations/' + annotation, 'w') as f:
                json.dump(array_annotations, f)
        array_annotations = []
        count += 1



