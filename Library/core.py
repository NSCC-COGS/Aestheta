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
import shapefile

from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.ensemble import GradientBoostingClassifier

# Local
try:
    from .model import Model
    from .placemark import Placemark
    from .tile_source import TileSource
except ImportError as error:
    from pathlib import Path
    
    this_file = Path(__file__).name
    raise ImportError(
        f'Running {this_file} directly is no longer supported. From now on,'
        f' go to the Aestheta directory and use "python -m Library.core"'
    ) from error

Placemark.load_config()
TileSource.load_config()

## function to read/load shapefiles based on file name
def shpreader(fname, show = False):
    shp = shapefile.Reader(fname)
    
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

def getTile(xyz=[0,0,0], source='google_map', show=False):
    source2, layer = TileSource.get_alias(source)
    tile = source2.tile(xyz, zoom=None, layer=layer)

    if show:
        plt.imshow(tile)
        plt.show()
        
    return tile
    
def getTiles_3x3(xyz=[0,0,0], source = 'google_map', show=False):
    source2, layer = TileSource.get_alias(source)
    tile = source2.tile_3x3(xyz, zoom=None, layer=layer)
    
    if show:
        plt.imshow(tile)
        plt.show()
        
    return tile

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
    
def saveModel(classModel, classes, sillyName=None):
    model = Model(model=classModel, classes=classes)
    model.save(label=sillyName)

def loadModel(name=None, model_dir='Models'):
    model = Model.load_named_or_latest(filename=name, dir_path=model_dir)
    return model.model, model.classes

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

if __name__ == '__main__':
    #for now we can put tests here!
    if 0: #test load the wms tile
        testTile = getTile()
        plt.imshow(testTile)
        plt.show()

    if 0: #test the simple classifier 
        img_RGB = getTile(source = 'google_sat')
        img_features = getTile(source = 'google_map')
        classModel,classes = simpleClassifier(img_RGB, img_features)
        img_class = classifyImage(img_RGB, classModel, classes)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img_RGB);ax1.axis('off');ax1.set_title('RGB')
        ax2.imshow(img_features);ax2.axis('off');ax2.set_title('Features')
        ax3.imshow(img_class);ax3.axis('off');ax3.set_title('Classification')
        plt.show()

    if 0: #test 3x3 tile
        xyz_novaScotia = [41,45,7]
        img_RGB = getTile(xyz = xyz_novaScotia, source = 'google_sat', show=True)
        img_RGB = getTiles_3x3(xyz = xyz_novaScotia, source = 'google_sat', show=True)
        input('press enter to continue with experimental version...')
        img_RGB = getTiles_experimental(xyz = xyz_novaScotia, source = 'google_sat', show=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img_RGB);ax1.axis('off');ax1.set_title('RGB')
        ax2.imshow(img_features);ax2.axis('off');ax2.set_title('Features')
        ax3.imshow(img_class);ax3.axis('off');ax3.set_title('Classification')
        plt.show()
    if 0: #test image shift difference
        img_RGB = getTile(xyz = [41,45,7], source = 'google_sat')
        image_shift_diff(img_RGB[:,:,0], show=True)
        img_RGB = getTile(xyz = [41,45,7], source = 'google_map')
        image_shift_diff(img_RGB[:,:,0], show=True)
        img_RGB = getTiles_3x3(xyz = [41,45,7], source = 'google_sat')
        image_shift_diff(img_RGB[:,:,0], show=True)

    if 0: # test image convolution 
        img_RGB = getTile(xyz = [41,45,7], source = 'google_sat')
        img = img_RGB[:,:,0]
        img_c = image_convolution(img)

        img_c = img_c*1.0 - img*1.0

        plt.imshow(img_c, cmap='gray')
        plt.show()

    if 0: #test image convolution RGB
        img_RGB = getTile(xyz = [41,45,7], source = 'google_sat')
        img_RGB_c = image_convolution_RGB(img_RGB)

        img_RGB_c = img_RGB_c*1.0 -  img_RGB*1.0

        plt.imshow(img_RGB_c)
        plt.show()
        
    if 0:
        # Coordinates of NSCC COGS; zoom level 15
        lat, lon, z = 44.88516350846845, -65.16834212839683, 15

        img_RGB = getTile(xyz=[lon, lat, z], source='google_sat')
        img = img_RGB[:,:,0]
        img_c = image_convolution(img)

        img_c = img_c*1.0 - img*1.0
        plt.imshow(img_c, cmap='gray')
        plt.show()

    if 0: #test model save
        img_RGB = getTile(source = 'google_sat')
        img_features = getTile(source = 'google_map')
        classModel,classes = simpleClassifier(img_RGB, img_features)
        print('Testing model save')
        saveModel(classModel, classes, sillyName = 'HelloEarth100')
        print('Model saved')

    if 0: #test load model
        img_RGB = getTile(source = 'google_sat')
        classModel, classes = loadModel(name='simpleClassifier_20210302180953_64_HelloEarth100.aist')
        print('Loaded model by name')
        classModel, classes = loadModel()
        print('Loaded model omitting name')
        img_class = classifyImage(img_RGB, classModel, classes)
        
        plt.imshow(img_class)
        plt.show()

    if 0: #test load model, multiple
        ''' this is unreasonably slow!''' 
        classModel, classes = loadModel()
        img_RGB_1 = getTile(source = 'google_sat')
        img_RGB_2 = getTile(xyz = [41,45,7], source = 'google_sat')
        img_class_1 = classifyImage(img_RGB_1, classModel, classes)
        img_class_2 = classifyImage(img_RGB_2, classModel, classes)
        
        fig, (ax1, ax2, ) = plt.subplots(1, 2)
        ax1.imshow(img_class_1);ax1.axis('off');ax1.set_title('Test_1')
        ax2.imshow(img_class_2);ax2.axis('off');ax2.set_title('Test_2')
        plt.show()

    if 0: #test default/recent model
        img_RGB = getTile(source = 'google_sat')
        img_class = classifyImage(img_RGB)

        plt.imshow(img_class)
        plt.show()

    if 0: #test 3x3 classifier 
        xyz_novaScotia = [41,45,7]
        img_RGB = getTiles_3x3(xyz = xyz_novaScotia, source = 'google_sat')
        img_class = classifyImage(img_RGB)

        plt.imshow(img_class)
        plt.show()
        
    if 0:
        # Test Placemark and TileSource. This uses relative imports, so you have
        # to run this with "python -m Library.core". "python Library/core.py"
        # will give you an error!
        
        # Use Google Maps as our TileSource.
        google_maps = TileSource.get('google_maps')
        
        # Use the Eiffel Tower as our Placemark.
        eiffel_tower = Placemark.get('eiffel_tower')
        
        # We can use aliases, too:
        if Placemark.get('eiffel') == eiffel_tower:
            print('Placemarks are identical.')
        
        # Feed the Eiffel Tower into Google Maps. Use the satellite layer.
        # The Eiffel tower has a default zoom level, so we don't have to specify
        # one.
        print(
            f'Getting tile for {eiffel_tower.description}. Default zoom level'
            f' is {eiffel_tower.default_zoom}.'
        )
        tile1 = google_maps.tile(eiffel_tower, layer='satellite')
        
        plt.imshow(tile1)
        plt.show()
        
        # Second way to do it: just supply the Placemark name/alias to the
        # tile() function/method, and the TileSource will look it up.
        #
        # In this case, we explicitly set the zoom level to override the
        # Placemark's default one.
        tile2 = google_maps.tile('cogs', zoom=9, layer='satellite')
        
        plt.imshow(tile2)
        plt.show()
        
        # getTile is nice in that you can specify a map source and layer with
        # one name, like google_map, google_sat, etc.
        #
        # TileSource lets you define aliases that can give you a source and
        # a layer.
        google_maps, layer = TileSource.get_alias('google_map')
        tile3 = google_maps.tile('eiffel', layer=layer)
        
        plt.imshow(tile3)
        plt.show()
        
        # This also works for MapSources that don't define layers.
        openstreetmap, layer = TileSource.get_alias('osm')
        tile4 = openstreetmap.tile_3x3('eiffel', layer=layer)
        
        plt.imshow(tile4)
        plt.show()
    if 1:
        getTile(xyz=[41, 45, 7], source='mapbox_sat', show=True)
        getTile(xyz=[-65.17, 44.89, 15], source='osm', show=True)
        getTile(xyz=['eiffel', 7], source='esri', show=True)
        getTile('eiffel', show=True)
        
        cogs = Placemark.get('cogs')
        getTile(cogs, show=True)