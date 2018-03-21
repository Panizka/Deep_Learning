import im_loader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import requests
import random
from pathlib import Path
import os
browser_headers = [
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:44.0) Gecko/20100101 Firefox/44.0"}
    ]
def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


# In[25]:


def download_image(im, folder, overwrite=False, headers=browser_headers):
    tag = im.get('tag')
    url = im['url']
    im_dir = os.path.join(folder, tag) if tag else folder    
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    im_path = os.path.join(im_dir, os.path.basename(url))
    if os.path.isfile(im_path) and not overwrite:
        print("File already exists...Skip!")
        #return 1
    else:
        print("Downloading ", url)
        r = requests.get(url, headers=random.choice(headers), verify=True)
        if ( r.status_code == requests.codes.ok and r.content ):
            open(im_path, 'wb').write(r.content)
        #return 0

def get_image_category(id):
    if id in test_images:
        return "test"
    elif id in train_images:
        return "train"
    elif id in validation_images:
        return "validation"
    else:
        return "none"


# In[27]:


def find(name,path):
    for root, dirs, files in os.walk(path):
        return os.path.join(root,name)
    return None


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

    
def parallel_process(array, function, n_jobs=10, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def download_target_classes(data_map, target_class_names):
    for item in target_class_names:
        # change test to train
        target_class_map = im_loader.filter_image_set(data_map,'train',item)
        urls = sorted(target_class_map.values())
        ids = sorted(target_class_map.keys())
        target_images_2_download = [{"im" : data_map['image_map'][i_id], "folder" : "downloads_test"} for i_id in ids]
        print("Downloading ", item)
        parallel_process(target_images_2_download, download_image, use_kwargs=True)
    return

