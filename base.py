import glob
import numpy as np
from PIL import Image
import time
import tracemalloc


# memory usage evaluate
def memory_usage(func_to_decorate):
    def the_wrapper():
        tracemalloc.start()
        func_to_decorate()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        counter = 0
        for stat in top_stats:
            counter += stat.size
        print('Memory used: ' + str(counter) + ' bytes')
    return the_wrapper


# calculation time evaluate
def calculation_time(func_to_decorate):
    def the_wrapper():
        start_time = time.time()
        func_to_decorate()
        finish_time = time.time()
        print('Output calculation time: ' + str(finish_time - start_time) + ' seconds')
    return the_wrapper


# input data array creating
def download_img(path, ext, pix_in_pic):
    dir = glob.glob(path + '\*.' + ext)
    counter = 1
    for i in dir:
        img = Image.open(str(i))
        img = img.convert('L')
        if dir.index(i) == 0:
            np_img = np.array(img, float)
            continue
        np_img = np.append(np_img, np.array(img, float), 0)
        counter += 1
    dataset = np.reshape(np_img, (counter, pix_in_pic))
    dataset /= 255
    return dataset


# custom switch
def switch(arg):
    if arg == 0:
        return 'circle'
    if arg == 1:
        return 'square'
    if arg == 2:
        return 'triangle'


# test image
def test_img(path):
    img = Image.open(path)
    img = img.convert('L')
    np_img = np.array(img, float)
    np_img /= 255
    dataset = np.reshape(np_img, (1, 49))
    return dataset


'''
# results array creating
def download_results(path, ext):
    dir = glob.glob(path + '\*.' + ext)
    for i in dir:
        path_arr = str(i).rsplit('\\')
        if dir.index(i) == 0:
            results = np.array(path_arr[-1].rsplit('_')[-2])
            continue
        results = np.append(results, np.array(path_arr[-1].rsplit('_')[-2]))
    return results
'''
