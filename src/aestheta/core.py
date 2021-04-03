# Standard Library
import math
import os
import pickle
import re
import urllib.request
import struct

from datetime import datetime

# Third-party
import imageio
import numpy as np
import requests
import shapefile # temporarily removed so our code works in colab!
import cv2

from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.ensemble import GradientBoostingClassifier

# Local
# Nothing here yet!

def features_from_image(img):
    features = img.reshape(-1, img.shape[2])
    return features

def image_from_features(features, width):
    length,depth = features.shape
    height = int(length/width)
    img = features.reshape(width,height,depth)
    return img

def unique_classes_from_image(img):
    features = features_from_image(img)
    classes,arr_classes,counts =  np.unique(features, axis=0, return_inverse=True, return_counts=True)
    return classes, arr_classes, counts

def histogram_from_image(img):
    classes, arr_classes, counts = unique_classes_from_image(img)
    bar_arr_dim = arr_classes.max()
    bar_arr = np.zeros((bar_arr_dim,bar_arr_dim,4)).astype(int)
    counts = np.log(counts)
    bar_heights = (counts/counts.max()*bar_arr_dim).astype(int)
    for i in range(bar_arr_dim):
        bar_arr[i,0:bar_heights[i],0:img.shape[2]]=classes[i]
        bar_arr[i,0:bar_heights[i],3]=255 #not that elegant, sets transparency 
    return bar_arr

def kmeans(img, k=3, show = True, iterations = 100):
    features = features_from_image(img)
    features = np.float32(features)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 0.2)
    _, labels, (centers) = cv2.kmeans(features, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    if show:
        plt.imshow(segmented_image)
        plt.show()
        return segmented_image
    else:
        return segmented_image

def ourPlot(a, interpolation = 'bilinear', histogram=True):
    stats = {
        'max' : np.nanmax(a),
        'min' : np.nanmin(a),
        'mean' : np.nanmean(a),
        'std' : np.nanstd(a),
        'bitDepth' : a.dtype,
        'dimensions' : a.shape,
        'top_left_value' : a[0,0]
    }

    for item in stats:
        print('%s: %s'%(item, stats[item]))

    plt.cla()
    plt.subplot(121)
    plt.imshow(a ,interpolation = interpolation)
    if histogram:
        plt.subplot(122)
        plt.hist(a.flatten(),bins=100)
        s0 = stats['mean'] - stats['std']
        s1 = stats['mean'] + stats['std']
        plt.axvline(s0,c='red')
        plt.axvline(s1,c='red')

## function to read/load shapefiles based on file name
#
# This won't work without the shapefile dependency, so I've commented it out.
# We can bring it back in once we've finished converting this to a Python
# package.
#

def shpreader(fname, show = False):
    shp = shapefile.Reader(fname) # note this currently wont work!
    
    # show if show is passed as true
    if show:
        plt.figure()
        
        for shape in shp.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            plt.plot(x,y)
        
        plt.show()
        
    # close the reader object and return it
    shp.close()
    return shp

# Adapted from deg2num at https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
def tile_from_coords(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return [tile_x, tile_y, zoom]

# Adapted from num2deg at https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2
def coords_from_tile(tile_x, tile_y, zoom):
    n = 2.0 ** zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return [lat_deg, lon_deg, zoom]

def getTile(xyz=[0,0,0], source='google_map', show=False):
    '''grabs a tile of a given xyz (or lon, lat, z) from various open WMS services
    note: these services are not meant to be web scraped and should not be accessed excessively'''

    # If our coords are floats, assume we're dealing with lat and long, and
    # convert them to tile x, y, z.
    x, y, z = xyz
    if isinstance(x, float) and isinstance(y, float):
        x, y, z = tile_from_coords(x, y, z)

    print(x, y, z)

    if source == 'google_map':
        url = f'http://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
    elif source == 'google_sat':
        url = f'http://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    elif source == 'osm_map':
        url = f'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
    elif source == 'mapbox_sat':
        TOKEN = 'pk.eyJ1Ijoicm5zcmciLCJhIjoiZTA0NmIwY2ZkYWJmMGZmMTAwNDYyNzdmYzkyODQyNDkifQ.djD5YCQzikYGFBo8pwiaNA'
        url = f'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={TOKEN}'
    elif source == 'esri':
        # otiles was down so replaced with esri - a nice source
        url = f'http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'
    elif source == 'wmf':
        # otiles was down so replaced with esri - a nice source
        url = f'http://c.tiles.wmflabs.org/osm-no-labels/{z}/{x}/{y}.png'

    #creates a header indicating a user browser to bypass blocking, note this is not meant for exhaustive usage
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
    res = requests.get(url, stream=True, headers=headers)
    img = imageio.imread(res.content)

    if show:
        plt.imshow(img)
        plt.show()
    else:
        return img

def simpleClassifier(img_RGB, img_features, subsample = 100):
    print('training  classifier...')
    classes,arr_classes =  np.unique(img_features.reshape(-1, img_features.shape[2]), axis=0, return_inverse=True)

    
    arr_RGB = img_RGB.reshape(-1, img_RGB.shape[-1])
    arr_RGB_subsample = arr_RGB[::subsample]
    arr_classes_subsample = arr_classes[::subsample]
    #classModel = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
    # max_depth=1, random_state=0,verbose=1).fit(arr_RGB_subsample, arr_classes_subsample)
    classModel = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=1,
            random_state=0,
            verbose=1
        ).fit(arr_RGB_subsample, arr_classes_subsample)

    return classModel, classes

def getDir(dirName = 'Models'):

    # this looks locally to this file and finds the dir based on the same
    model_dir = os.path.dirname(__file__)
    model_dir = os.path.join(model_dir,dirName)

    return model_dir
    
def saveModel(classModel, classes, sillyName = None, model_dir = None):

    if model_dir == None:
        model_dir = getDir('Models')

    #puts the classificaion model and the classes into a list 
    model = [classModel, classes]

    #creates a string for the current time 
    now = datetime.now()
    uniqueString = now.strftime("%Y%m%d%H%M%S") #https://www.programiz.com/python-programming/datetime/strftime
    
    python_bits_user = struct.calcsize("P") * 8 #this will print 32 or 64 depending on python version
    uniqueString += '_'+str(python_bits_user)

    if sillyName:
        uniqueString += '_'+sillyName

    #saves out the model list with a name from the current time
    current_model = f'simpleClassifier_{uniqueString}.aist'
    filename = os.path.join(model_dir,current_model)

    print('saving model to',filename)
    pickle.dump(model, open(filename, 'wb'))
    print('complete..')

def loadModel(name = None, model_dir = None):

    if model_dir == None:
        model_dir = getDir('Models')

    model_list = os.listdir(model_dir)
    print(model_list)

    if name == None: #loads most recent
        maxDate = 0 
        newest_model = None
        print('getting most recent model')
        for model_name in model_list:
            model_name_hack = model_name.replace('.','_')
            model_name_list = model_name_hack.split('_')
            date_time = int(model_name_list[1])

            python_bits_user = struct.calcsize("P") * 8 #this will print 32 or 64 depending on python version
            python_bits_file = int(model_name_list[2])

            if date_time > maxDate and python_bits_user == python_bits_file:
                newest_model = model_name
                maxDate = date_time

            print(date_time)
            # print(model_name.split('_'))
            # a = re.split('_|.',model_name)
            # print(a)

    try:        
        filename = os.path.join(model_dir,newest_model)
        print(filename)
    except:
        print(f'No Model found for {python_bits_user} bit pythion')
        filename = input("enter model path...")
        return

    classModel, classes = pickle.load(open(filename, 'rb'))

    return classModel, classes

def classifyImage(img_RGB,classModel = None ,classes = None):
    if not classModel: # if no model set
        try:
            classModel,classes = loadModel() #loads the most recent model
        except:
            print('no model found')
            return 

    print('applying classification...')
    # Getting shape of the incoming file
    arr_RGB_shape = img_RGB.shape

    if 1: # a very temporary fix, the resultant array needs the 'depth' of the classes (which is RGBA)
        arr_RGB_shape = list(arr_RGB_shape) #https://www.w3schools.com/python/gloss_python_change_tuple_item.asp
        arr_RGB_shape[2] = classes.shape[1]  

    arr_RGB = img_RGB.reshape(-1, img_RGB.shape[-1])
    arr_classes_model = classModel.predict(arr_RGB)
    arr_label_model = classes[arr_classes_model]
    # Substituting the shape of the incoming file arr_RGB_shape instead of a hard coded 256x256 size
    img_class = np.reshape(arr_label_model,arr_RGB_shape) #hard coded for 256x256 images!
    
    return img_class
  
#def classifyImage(img_RGB,classModel,classes):
#    print('applying classification...')
#    arr_RGB_shape = img_RGB.shape
#    arr_RGB = img_RGB.reshape(-1, img_RGB.shape[-1])
#    arr_classes_model = classModel.predict(arr_RGB)
#    arr_label_model = classes[arr_classes_model]
#    img_class = np.reshape(arr_label_model,arr_RGB_shape) #hard coded for 256x256 images!
#
#    return img_class

def getTiles_3x3(xyz=[0,0,0], source = 'google_map', show=False):
    x,y,z = xyz

    # check if input are coordinates (float)
    if isinstance(x, float) and isinstance(y, float):
        x, y, z = tile_from_coords(x, y, z)

    idx = [-1,0,1]
    # idx = [-2-1,0,1,2]
    img = 'Start'
    for j in idx:
        for k in idx:
            print(j,k)
            tile_img = getTile(xyz=[x+j,y+k,z], source = source, show=False)
            #print(f"tile image shape {tile_img.shape}")
            if img == 'Start':
                img = np.zeros((tile_img.shape[0]*3,tile_img.shape[1]*3,tile_img.shape[2]),dtype=tile_img.dtype) 
            x0 = (j+1)*tile_img.shape[0]
            y0 = (k+1)*tile_img.shape[1]
            img[y0:y0+tile_img.shape[0],x0:x0+tile_img.shape[1]] = tile_img
 
    if show:
        plt.imshow(img)
        plt.show()

    else:
        return img

def getTiles_experimental(xyz=[0,0,0], source = 'google_map', show=False):
    x,y,z = xyz
    idx = [-1,0,1]
    tiles = []
    img = 'Start'
    for j in idx:
        for k in idx:
            print(j,k)
            # tiles.append(getTile(xyz=[x+j,y+k,z], source = source, show=False)*0+100-k*100)
            tiles.append(getTile(xyz=[x+j,y-k,z], source = source, show=False))

    tiles = np.array(tiles)
    print(tiles.shape)

    plt.imshow(tiles[0])
    plt.show()

    img = tiles.reshape(3*256*3*256,3)
    print(img.shape)
    
    if show:
        plt.imshow(img)
        plt.show()

    else:
        return img

# Testing Nomalising Difference - 21/03/21
# This function still has a few errors, so I've commented it out for now.

def norm_diff(img_RGB, B1, B2, show = False):
    # get true band numbers (Bands 1 and 2 are index 0 and 1)
    B1 = B1 - 1
    B2 = B2 - 1

    # test if band selection was valid
    if B1 in range(0,3) and B2 in range(0,3):
        # get bands from tile
        img_B1 = img_RGB[:,:,(B1)]
        img_B2 = img_RGB[:,:,(B2)]
        # convert to float32
        img_B1 = np.float32(img_B1)
        img_B2 = np.float32(img_B2)
        #calculate normalized difference
        ndiff = (img_B1 - img_B2) / (img_B1 + img_B2)
        
        # plot with matplotlib if uses wants      
        if show:
            plt.imshow(ndiff)
            plt.show()
        else:
            return ndiff
    # show user error of they selected bands out of range
    else:
        print("Select bands between 1 and 3")



def image_shift_diff(img_RGB, show=False, axis=0, distance = 1):
    img_shifted = np.roll(img_RGB,distance,axis=axis)
    img = img_shifted*1.0 - img_RGB*1.0 #multiplying by 1.0 is a lazy way yo convert an array to float

    if show:
        plt.imshow(img, cmap='gray')
        plt.show()

    else:
        return img

def image_convolution(img):
    kernel = np.array([
        [0,.125,0],
        [.125,.5,.125],
        [0,.125,0]])
    return ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    
def image_convolution_RGB(img_RGB):
    
    img_RGB = img_RGB * 1.0

    for band in range(0,img_RGB.shape[2]):

        img_RGB[:,:,band] = image_convolution(img_RGB[:,:,band])

    return img_RGB
