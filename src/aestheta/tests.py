if __name__ == '__main__':
    from core import *

    if 1: #test image to features and back
        testTile = getTile()
        testFeatures = features_from_image(testTile)
        print(testFeatures)

        testTile_2 = image_from_features(testFeatures, testTile.shape[0])
        plt.imshow(testTile_2)
        plt.show()


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
    
    if 0: #test image normalising difference
        
        img_RGB = getTile([44.6488,-63.5752,2],source='google_sat') #swapped lat/lon
        ND = norm_diff(img_RGB,  B1=1, B2=2, show=True)
        

        
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
        
    if 0: # Test image convolution with coordinate transforms.
        # Coordinates of NSCC COGS; zoom level 15
        lat, lon, z = 44.88516350846845, -65.16834212839683, 15
        
        # Convert coords -> tile -> coords
        x1, y1, z1 = tile_from_coords(lon, lat, z)
        x2, y2, z2 = coords_from_tile(x1, y1, z1)
        
        # Feed resulting lon, lat, z to getTile.
        img_RGB = getTile(xyz = [x2, y2, z2], source = 'google_sat')
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
        
    if 0: #not currently working
        # Test Placemark and TileSource. This uses relative imports, so you have
        # to run this with "python -m Library.core". "python Library/core.py"
        # will give you an error!
        from .placemark import Placemark
        from .tile_source import TileSource
        
        # Need to load configs first…
        Placemark.load_config()
        TileSource.load_config()
        
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
