import csv
import json
import itertools
from concurrent.futures import ProcessPoolExecutor
import requests
import random
import os
import re
from collections import defaultdict
from pathlib import Path
from itertools import groupby
from matplotlib.patches import Rectangle
from PIL import Image
import cProfile

def load_classes(class_description, class_file):
    #class_file = path to classes.txt
    with open (class_file, 'r') as f:
        for line in f:
            class_code = line.strip()
            #if class_code not in desp_keys:
            class_description[class_code] = {"id": class_code}
    return class_description

 
# In[3]:


def load_class_description(class_map, class_description_file):
    #class_map = {}
    with open(class_description_file, 'r') as f:
        for line in f:
            #print(line)
            class_code, class_name = line.split(',',1)
            key = class_code.strip()
            val = class_name.strip()
            entry = class_map.get(key)
            if entry:
                entry["class-name"] = val
            else:
                class_map[key] = {"class-name":val}
            #print(line)
    return class_map


# In[4]:


def query_class_info(class_map):
    class_names = []
    for k, v in class_map.items():
        class_names.append(k)
    return class_names


# In[5]:


def load_class_type(class_map, class_file, tag):
    with open(class_file, 'r') as f:
        for line in f:
            class_code = line.strip()
            v = class_map.get(class_code)
            if v:
                v[tag] = True
            else:
                class_map[class_code] = {tag: False}
    return class_map

def load_images(image_map, image_path):
    with open(image_path, 'r') as f:
        next (f)
        for line in f:
            image_entry = line.split(',')
            image_map[image_entry[0]] = {"train": True, "url": image_entry[2].strip(), "Title": image_entry[7].strip(), "OriginalSize": image_entry[8].strip()}
    return image_map 


# In[9]:


def load_image_type(image_map, image_path, tag):
    new_keys = []
    with open(image_path, 'r') as f:
        next (f)
        for line in f:
            image_entry = line.split(',')
            key = image_entry[0]
            v = image_map.get(key)
            new_keys.append(key)
            #print(image_entry[0])
            #print(v)
            if v:
                image_map[image_entry[0]]["tag"] = tag;
            else:
                image_map[image_entry[0]] = {"tag" : tag, "id" : key, "url": image_entry[2].strip(), "Title": image_entry[7].strip(), "OriginalSize": image_entry[8].strip()}
    return image_map, set(new_keys)

def add_annotation(image_map, annotation_file):
    with open(annotation_file, 'r') as f:
        next(f)
        image_ids = set()
        for row in csv.reader(f):
            image_id, class_id = row[0], row[2]
            image_ids.add(image_id)
            key = row[0] + row[2] + ''.join(row[4:7])
            new_anno = {"label_source" : row[1], 
                        "label_confidence" : float(row[3]), 
                        "class_id" : class_id,
                        "coordinate" : [[float(row[4]), float(row[5])], #Xmin, Xmax
                                        [float(row[6]), float(row[7])]]} #Ymin, Ymax
            anno_field = "annotation"
            if not image_map.get(image_id):
                image_map[image_id] = {"id" : row[0], anno_field : {key: new_anno}}
            elif not image_map[image_id].get(anno_field):
                image_map[image_id][anno_field] = {key : new_anno}
            else:
                image_map[image_id][anno_field].update({key : new_anno})
    return image_map, image_ids

#def show_annotation_types(image_map, class_map, image_id):
 #   class_counts = defaultdict(int)
  #  for k, v in image_map[image_id]["annotation"].items():
   #     class_counts[class_map[v['class_id']]['class-name']] += 1
    #return class_counts 

def show_image_annotation_types(data_map, image_id):
    new_map = {}
    image_map = data_map['image_map']
    class_map = data_map['class_map']
    if image_map[image_id].get("annotation"):
        for k, v in image_map[image_id]["annotation"].items():
            class_id = v['class_id']
            if new_map.get(class_id):
                new_map[class_id][1] += 1
            else:
                new_map[class_id] = [class_map[class_id]['class-name'],1, 1]
        return new_map
    else:
        return {}

def show_set_annotation_types(data_map, my_image_set):
    image_map = data_map['image_map']
    class_map = data_map['class_map']
    image_set = data_map['image_sets'][my_image_set]
    image_list = list(image_set)
    new_map = show_image_annotation_types(data_map, image_list[0])
    for image_id in image_list[1:]:
        temp_map = show_image_annotation_types(data_map, image_id)
        for k,v in temp_map.items():
            if new_map.get(k):
                new_map[k][1] += v[1]
                new_map[k][2] += 1
            else:
                new_map[k] = v
    return new_map

def filter_image_set(data_map, image_set_name, target_class_name):
    image_map = data_map['image_map']
    class_map = data_map['class_map']
    image_set = data_map['image_sets'][image_set_name]
    temp_map = {}
    for image_id in image_set:
        if image_map[image_id].get("annotation"):
            for k, v in image_map[image_id]["annotation"].items():
                class_id = v['class_id']                
                class_name = class_map[class_id]['class-name']
                if class_name in target_class_name:
                    if temp_map.get(image_id):
                        temp_v = "hi"
                        #temp_map[image_id][1].update([class_name])
                        #print(class_id)
                        #print(image_map[image_id].get("url"))
                    else:
                        temp_map[image_id] = image_map[image_id].get("url")
    return temp_map

def show_image_bboxes(image_map, image_id, download_folder="/home/karbasi@ecs.baylor.edu/repos/Machine_Learning/downloads/"):
    image_map_entry = image_map[image_id]
    cat = get_image_category(image_id)
    #name = find(os.path.basename(image_map_entry['url']), download_folder+cat)
    name = download_image(image_map_entry['url'], os.path.join(download_folder, cat))   
    im = np.array(Image.open(download_folder+cat+"/"+name))
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    height, width = im.shape[:2]
    if image_map_entry.get("annotation"):
        for k, v in image_map_entry.get("annotation").items():
            coordinate = v['coordinate']
            xMin = coordinate[0][0] * width
            xMax = coordinate[0][1] * width
            yMin = coordinate[1][0] * height
            yMax = coordinate[1][1] * height
            rect = patches.Rectangle((xMin, yMin), (xMax - xMin), (yMax - yMin), linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
    plt.show()

def load_data_map(data_folder):
    data_map = {}
    print("loading class map...")
    class_map = load_classes({}, os.path.join(data_folder,"classes/classes.txt"))
    class_map = load_class_description(class_map, os.path.join(data_folder,"classes/class-descriptions.csv"))
    class_map = load_class_type(class_map, os.path.join(data_folder,"classes/classes-trainable.txt"), "trainable")
    class_map = load_class_type(class_map, os.path.join(data_folder,"classes/classes-bbox-trainable.txt"), "bbox-trainable")
    class_map = load_class_type(class_map, os.path.join(data_folder,"classes/classes-bbox.txt"), "bbox")
    

    print("loading image map...")
    image_map, train_images = load_image_type({}, os.path.join(data_folder,"images/train/images.csv"), "train")
    image_map, test_images = load_image_type(image_map, os.path.join(data_folder,"images/test/images.csv"), "test")
    image_map, validation_images = load_image_type(image_map, os.path.join(data_folder,"images/validation/images.csv"), "validation")
    image_map, annotation_test_set = add_annotation(image_map, os.path.join(data_folder,"bbox_annotations/test/annotations-human-bbox.csv"))
    image_map, annotation_train_set = add_annotation(image_map, os.path.join(data_folder,"bbox_annotations/train/annotations-human-bbox.csv"))

    # print("Creating class name map")
    # class_name_test = set()
    # for image_id in test_images:
    #     if image_map[image_id].get("annotation"):            
    #         for k, v in image_map[image_id]["annotation"].items():
    #             class_id = v['class_id']                
    #             class_name = class_map[class_id]['class-name']
    #             class_name_test.add(class_name)

    # class_name_train = set()
    # for image_id in train_images:
    #     if image_map[image_id].get("annotation"):            
    #         for k, v in image_map[image_id]["annotation"].items():
    #             class_id = v['class_id']                
    #             class_name = class_map[class_id]['class-name']
    #             class_name_test.add(class_name)

                
                

    
    bbox_map = {}
    for k, v in class_map.items():
        if v.get('bbox'):
            bbox_map[k] = v

    bbox_trainable_map = {}
    for k, v in class_map.items():
        if v.get('bbox-trainable'):
            bbox_trainable_map[k] = v

    trainable_map = {}
    for k, v in class_map.items():
        if v.get('trainable'):
            trainable_map[k] = v

    print("loading data map...")
    data_map['class_map'] = class_map
    data_map['image_map'] = image_map
    data_map['image_sets'] = {}
    data_map['image_sets']['test'] = test_images
    data_map['image_sets']['train'] = train_images
    data_map['image_sets']['validation'] = validation_images
    data_map['image_sets']['annotation_test_images'] = annotation_test_set
    data_map['image_sets']['annotation_train_images'] = annotation_train_set
    data_map['class_types'] = {}
    data_map['class_types']['bbox'] = bbox_map
    data_map['class_types']['bbox-trainable'] = bbox_trainable_map
    data_map['class_types']['trainable'] = trainable_map

    return data_map
