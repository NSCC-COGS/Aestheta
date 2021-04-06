


from core import *
import pandas as pd

sceneList = r'https://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz'

def getScene(url, show=False):
    #creates a header indicating a user browser to bypass blocking, note this is not meant for exhaustive usage
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
    res = requests.get(url, stream=True, headers=headers)
    img = imageio.imread(res.content)

    if show:
        plt.imshow(img)
        plt.show()
    else:
        return img

def loadSceneList(sceneList, sceneTest=None, cloudCoverMax=None):
    #using pandas, read the scene list into a dataframe object
    sceneDf = pd.read_csv(sceneList , nrows=sceneTest)

    #set the aquasition date field as a datetime object
    sceneDf.acquisitionDate = pd.to_datetime(sceneDf.acquisitionDate)

    #print some example information for
    print(sceneDf.head())
    print(sceneDf['acquisitionDate'])
    print(sceneDf.columns)
    if cloudCoverMax:
        #lets filter the dataset down to just clouds less than a given %
        cloudFilter = sceneDf['cloudCover'] < cloudCoverMax
        sceneDf = sceneDf[cloudFilter]
    return sceneDf

def findPathRow(sceneDf, lat, lon):
    #calculate the distance from our LatLon to each scene
    sceneDf['lat'] = (sceneDf['min_lat'] + sceneDf['max_lat']) / 2.0
    sceneDf['lon'] = (sceneDf['min_lon'] + sceneDf['max_lon']) / 2.0
    sceneDf['dist'] = np.sqrt(( sceneDf['lat']-lat)**2 + (sceneDf['lon'] - lon)**2)

    #now lets find the nearest path/row
    nearestFilter = sceneDf['dist'].idxmin()
    nearestPath = sceneDf['path'][nearestFilter]
    nearestRow = sceneDf['row'][nearestFilter]

    return nearestPath, nearestRow

def selectScene(sceneDf, path, row):
    #reduce the avilable scenes down to the nearest path/row
    sceneDf_nearest = sceneDf[(sceneDf['path']==path) & (sceneDf['row']==row)]

    #now find the newest capture in the nearest path/row
    newestFilter = sceneDf_nearest['acquisitionDate'].idxmax()

    # report the final selected scene
    selectedScene = sceneDf_nearest.loc[newestFilter]

    return selectedScene

def getImageIO(selectedScene, band, imageDir=None):
    # report the URL for the image index
    indexUrl = selectedScene['download_url']
    productId = selectedScene['productId']

    url = os.path.split(indexUrl)[0]
    imageName = productId + ('_B%i.TIF') % band
    
    imageUrl = r'/'.join([url,imageName])

    if imageDir:
        imagePath = os.path.join(imageDir,imageName)

        print(imageUrl, imagePath)
        return imageUrl,imagePath

    else:
        return imageUrl

def downloadImage(imageUrl,imagePath):

    if not os.path.exists(imagePath):
        print('downloading...', imageUrl)
        response = urllib.request.urlopen(imageUrl)
        imageFile  = response.read()
        imageDir = os.path.split(imagePath)[0]
        if not os.path.exists(imageDir):os.makedirs(imageDir)
        with open(imagePath, 'wb') as f:
            f.write(imageFile)
    else:
        print(imagePath, 'already found')

def plotResults(lat,lon, selectedScene,sceneDf,imagePath):
    #prepare a plot
    plt.figure()

    # assign plotting area to 2x1 subplot, starting with area 1
    plt.subplot(211)

    #reduce the total captures to a maximum to speed up  plotting 
    try:
        plotDf = sceneDf[0:10000]
    except:
        plotDf = sceneDf

    #plot the reduced set of all image captures
    # plt.scatter(plotDf['lon'],plotDf['lat'], c='blue', linewidth = 0, marker ='.', alpha = .1)
    plt.scatter(plotDf['lon'],plotDf['lat'],c=plotDf['acquisitionDate'].astype(np.int64), cmap = 'viridis', marker ='.', alpha = .5)

    # plot the selected capture location
    plt.scatter(selectedScene['lon'],selectedScene['lat'], linewidth =5, marker = 'o', c='red')

    # plot the input lat/lon
    plt.scatter(lon,lat, marker = 'x', c='black')

    ''' some other plots include:
    # plt.scatter(plotDf.mean_lon,plotDf.mean_lat,c=plotDf.acquisitionDate.astype(np.int64), cmap = 'viridis', marker ='.', alpha = .5)
    # plt.scatter(sceneDf_nearest.mean_lon,sceneDf_nearest.mean_lat, linewidth =5, marker = 'o', c='red')
    '''

    image = tifffile.imread(imagePath)

    print(image.shape)
    plt.subplot(212)
    imagePreview = image[::10,::10]
    plt.imshow(imagePreview, cmap='bone')

    plt.show()   

###############################################

def makeCompositeRGBA(selectedScene,bands=[4,3,2]):
    import tifffile
    import image_functions
    images = []
    for band in bands:
        imageUrl, imagePath = getImageIO(selectedScene, band, imageDir)
        downloadImage(imageUrl,imagePath)
        im = tifffile.imread(imagePath)[::1,::1] # this will reduce the size of the arrays by a factor of 10x10 to save on memory
        im = image_functions.convert32(im, inplace=True)
        im = image_functions.stdScale(im, n=.5, inplace=True) # note that I did play with the number of standard deviation here
        im = image_functions.normalize(im, inplace=True)
        im,mask = image_functions.scale8bit(im, return_mask=True, inplace=True)
        images.append(im)
    images.append(mask)

    return np.dstack(images)

###########################

def getLandsatTile(xyz=[44.88,-65.16,10], band=2):
    lat,lon,zoom = xyz
    band = band
    tileCoords = tile_from_coords(lat,lon,zoom)
    img = getTile(tileCoords, source='google_sat') 
    if 1:
        # plt.imshow(img)
        # plt.show()



        sceneDf = loadSceneList(sceneList, sceneTest=None, cloudCoverMax = 2)
        path,row = findPathRow(sceneDf, lat, lon)
        selectedScene = selectScene(sceneDf, path, row)
        imageUrl = getImageIO(selectedScene, band)

        print(imageUrl)

    if 0:
        imageUrl = r'https://s3-us-west-2.amazonaws.com/landsat-pds/c1/L8/009/029/LC08_L1TP_009029_20210111_20210111_01_RT/LC08_L1TP_009029_20210111_20210111_01_RT_B2.TIF'
       
    # https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html?highlight=tile%20map%20service#virtual-warping

    from affine import Affine
    import mercantile

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(imageUrl) as src:
        with WarpedVRT(src, crs='EPSG:3857',
                    resampling=Resampling.bilinear) as vrt:

            # Determine the destination tile and its mercator bounds using
            # functions from the mercantile module.
            '''
            dst_tile = mercantile.tile(*vrt.lnglat(), 9)
            left, bottom, right, top = mercantile.xy_bounds(*dst_tile)
            '''
            left, bottom, right, top = mercantile.xy_bounds(tileCoords)
            


            print(left, bottom, right, top)
            # input()

            # Determine the window to use in reading from the dataset.
            dst_window = vrt.window(left, bottom, right, top)

            # Read into a 3 x 512 x 512 array. Our output tile will be
            # 512 wide x 512 tall.
            data = vrt.read(window=dst_window, out_shape=(256, 256))

            # Use the source's profile as a template for our output file.
            profile = vrt.profile
            profile['width'] = 256
            profile['height'] = 256
            profile['driver'] = 'GTiff'

            # We need determine the appropriate affine transformation matrix
            # for the dataset read window and then scale it by the dimensions
            # of the output array.
            dst_transform = vrt.window_transform(dst_window)
            scaling = Affine.scale(dst_window.height / 256,
                                dst_window.width / 256)
            dst_transform *= scaling
            profile['transform'] = dst_transform

            fig, axs = plt.subplots(2)

            axs[0].imshow(data[0,:,:])
            axs[1].imshow(img)
            plt.show()

            # # Write the image tile to disk.
            # with rasterio.open('/tmp/test-tile.tif', 'w', **profile) as dst:
            #     dst.write(data)


def getLandsat(xyz=[0,0,0], source='nasa', show=False):
    '''grabs a tile of a given xyz (or lon, lat, z) from various open WMS services
    note: these services are not meant to be web scraped and should not be accessed excessively'''

    # If our coords are floats, assume we're dealing with lat and long, and
    # convert them to tile x, y, z.
    x, y, z = xyz
    # if isinstance(x, float) and isinstance(y, float):
    #     x, y, z = tile_from_coords(x, y, z)   
        
    if isinstance(x, int) and isinstance(y, int):
        x, y, z = coords_from_tile(x, y, z)

    print(x, y, z)

    lon = y
    lat = x
    # date = '2014-02-01'
    date = '2021-04-01'
    dim = '0.10' 
    TOKEN = 'DEMO_KEY'

    if source == 'nasa':
        # https://api.nasa.gov/
        # url = f'https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&date={date}&api_key={TOKEN}'
        url = f'https://api.nasa.gov/planetary/earth/imagery?lon={lon}&lat={lat}&date={date}&&dim={dim}&api_key={TOKEN}'

    #creates a header indicating a user browser to bypass blocking, note this is not meant for exhaustive usage
    headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
    
    print(url)
    
    res = requests.get(url, stream=True, headers=headers)
    img = imageio.imread(res.content)

    print(img.shape)

    if show:
        plt.imshow(img)
        plt.show()
    else:
        return img

if __name__ == '__main__':
    if 0: # test loading scene list
        testScenes = loadSceneList(sceneList,sceneTest=10)
        print(testScenes)

    if 0: # test downlading scene
        lat = 44.88
        lon = -65.16
        band = 1
        cloudCoverMax = 2
        # sceneList = "scene_list.txt"
        sceneTest = None # set a small number to test, or None
        imageDir = r'.\landsat'

        sceneDf = loadSceneList(sceneList, sceneTest, cloudCoverMax)

        # sceneDf = loadSceneList(sceneList)

        path,row = findPathRow(sceneDf, lat, lon)

        # print(path, row)

        selectedScene = selectScene(sceneDf, path, row)

        landsat_RGBA = makeCompositeRGBA(selectedScene,[4,3,2])
        ourPlot(landsat_RGBA)
        plt.show()
    
    if 0: #test getScene
        url = r'https://s3-us-west-2.amazonaws.com/landsat-pds/c1/L8/009/029/LC08_L1TP_009029_20210111_20210111_01_RT/LC08_L1TP_009029_20210111_20210111_01_RT_B2.TIF'
        testScene = getScene(url)
        plt.imshow(testScene)
        plt.show()

    if 0: #test rasterio read
        url = r'https://s3-us-west-2.amazonaws.com/landsat-pds/c1/L8/009/029/LC08_L1TP_009029_20210111_20210111_01_RT/LC08_L1TP_009029_20210111_20210111_01_RT_B2.TIF'
        src = rasterio.open(url)
        print(src.crs)
        print(src.crs.wkt)
        print(src.transform)
        plt.imshow(src.read(1)[::4,::4],vmin=-10)
        plt.show()

    if 0: #test getLandsatTile
        getLandsatTile()

    if 1: #test getLandsat
        # getData(xyz=[45.5,-63.5,0], source='nasa', show=True)
        getLandsat(xyz=[44.88514830522117,-65.1684601455184,0], source='nasa', show=True)