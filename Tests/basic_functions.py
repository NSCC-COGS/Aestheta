from Aestheta.Library.core import *

#test load the wms tile
def test_load_wms_file():
    from matplotlib import pyplot as plt
    testTile = getTile()
    plt.imshow(testTile)
    plt.show()

#test the simple classifier 
def test_simple_classifier():
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