'''import numpy as np
import urllib.request
import os
import requests
import imageio'''

def getTile(xyz=[0,0,0], source = 'google_map', show=False):
    '''grabs a tile of a given xyz from various open WMS services
    note: these services are not meant to be web scraped and should not be accessed excessively'''
    import requests,imageio

    #converts the list of xyz to variables  
    x,y,z = xyz
    print(x,y,z)

    #makes our WMS url from 
    if source == 'google_map':
        url = f'http://mt.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
    
    elif source == 'google_sat':
        url = f'http://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

    elif source == 'osm_map':
        url = f'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'

    elif source == 'mapbox_sat':
        TOKEN = 'pk.eyJ1Ijoicm5zcmciLCJhIjoiZTA0NmIwY2ZkYWJmMGZmMTAwNDYyNzdmYzkyODQyNDkifQ.djD5YCQzikYGFBo8pwiaNA'
        url = f'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={TOKEN}'

    elif source == 'otile':
        #note this may not work
        url = f'http://otile1.mqcdn.com/tiles/1.0.0/osm/{z}/{x}/{y}.png'


    #creates a header indicating a user browser to bypass blocking, note this is not meant for exhaustive usage
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}


    res= requests.get(url, stream = True, headers=headers)

    img = imageio.imread(res.content)

    if show:
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    else:
        return img

def simpleClassifier(img_RGB, img_features, subsample = 100):
    import numpy as np
    print('training  classifier...')
    classes,arr_classes =  np.unique(img_features.reshape(-1, img_features.shape[2]), axis=0, return_inverse=True)

    from sklearn.ensemble import GradientBoostingClassifier
    arr_RGB = img_RGB.reshape(-1, img_RGB.shape[-1])
    arr_RGB_subsample = arr_RGB[::subsample]
    arr_classes_subsample = arr_classes[::subsample]
    classModel = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,
                                    max_depth=1, random_state=0,verbose=1).fit(arr_RGB_subsample, arr_classes_subsample)

    return classModel, classes

def classifyImage(img_RGB,classModel,classes):
    import numpy as np
    print('applying classification...')
    # Getting shape of the incoming file
    arr_RGB_shape = img_RGB.shape
    arr_RGB = img_RGB.reshape(-1, img_RGB.shape[-1])
    arr_classes_model = classModel.predict(arr_RGB)
    arr_label_model = classes[arr_classes_model]
    # Substituting the shape of the incoming file arr_RGB_shape instead of a hard coded 256x256 size
    img_class = np.reshape(arr_label_model,arr_RGB_shape) #hard coded for 256x256 images!

    return img_class

def getTiles_3x3(xyz=[0,0,0], source = 'google_map', show=False):
    import numpy as np
    x,y,z = xyz
    idx = [-1,0,1]
    # idx = [-2-1,0,1,2]
    img = 'Start'
    for j in idx:
        for k in idx:
            print(j,k)
            tile_img = getTile(xyz=[x+j,y+k,z], source = source, show=False)
            if img == 'Start':
                img = np.zeros((256*3,256*3,3),dtype=tile_img.dtype) # this will certainly throw an error when input images are more than 3 bands
            x0 = (j+1)*256
            y0 = (k+1)*256
            img[y0:y0+256,x0:x0+256] = tile_img
 
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    else:
        return img

def getTiles_experimental(xyz=[0,0,0], source = 'google_map', show=False):
    import numpy as np
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

    from matplotlib import pyplot as plt
    plt.imshow(tiles[0])
    plt.show()

    img = tiles.reshape(3*256*3*256,3)
    print(img.shape)
    
    if show:
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    else:
        return img

def image_shift_diff(img_RGB, show=False, axis=0, distance = 1):
    import numpy as np
    img_shifted = np.roll(img_RGB,distance,axis=axis)
    img = img_shifted*1.0 - img_RGB*1.0 # what is happening to values under 0?

    if show:
        from matplotlib import pyplot as plt
        plt.imshow(img, cmap='gray')
        plt.show()

    else:
        return img

if __name__ == '__main__':
    #for now we can put tests here!
    if 0: #test load the wms tile

        from matplotlib import pyplot as plt
        testTile = getTile()
        plt.imshow(testTile)
        plt.show()

    if 0: #test the simple classifier 
        img_RGB = getTile(source = 'google_sat')
        img_features = getTile(source = 'google_map')
        classModel,classes = simpleClassifier(img_RGB, img_features)
        img_class = classifyImage(img_RGB, classModel, classes)

        from matplotlib import pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img_RGB);ax1.axis('off');ax1.set_title('RGB')
        ax2.imshow(img_features);ax2.axis('off');ax2.set_title('Features')
        ax3.imshow(img_class);ax3.axis('off');ax3.set_title('Classification')
        plt.show()

    if 0: #test 3x3 tile
        xyz_novaScotia = [41,45,7]
        img_RGB = getTiles_3x3(xyz = xyz_novaScotia, source = 'google_sat', show=True)
        input('press enter to continue with experimental version...')
        img_RGB = getTiles_experimental(xyz = xyz_novaScotia, source = 'google_sat', show=True)

    if 1: #test image shift difference
        img_RGB = getTile(xyz = [41,45,7], source = 'google_sat')
        image_shift_diff(img_RGB[:,:,0], show=True)
        img_RGB = getTile(xyz = [41,45,7], source = 'google_map')
        image_shift_diff(img_RGB[:,:,0], show=True)
        img_RGB = getTiles_3x3(xyz = [41,45,7], source = 'google_sat')
        image_shift_diff(img_RGB[:,:,0], show=True)

