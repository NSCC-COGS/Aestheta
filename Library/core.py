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

if __name__ == '__main__':
    #for now we can put tests here!
    from matplotlib import pyplot as plt
    testTile = getTile()
    plt.imshow(testTile)
    plt.show()