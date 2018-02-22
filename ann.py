from __future__ import absolute_import, division, print_function
from annoy import AnnoyIndex
from scipy import spatial
from tqdm import tqdm
from sklearn.externals import joblib 
import glob, os
import numpy as np
import img2vec
import matplotlib.pyplot as plt; plt.switch_backend('Qt5Agg')
import matplotlib.image as mpimg
import pickle
import sys
from config import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.dont_write_bytecode=True

if AUTO_ENCODER_ENABLE and not (os.path.isfile(AUTO_ENCODER_FILE) and os.path.isfile(MIN_MAX_PICKLE) and os.path.isfile(AUTO_ENCODED_VECTORS)):
  raise Exception('Please run generate autoencoder files by running "python autoencoder.py"')

# Load the Image Vectors or Auto-encoded vectors
loaded = np.load(IMAGE_VECTORS)
if AUTO_ENCODER_ENABLE:
  loaded2 = np.load(AUTO_ENCODED_VECTORS)
  X=loaded2['X']
else:
  X=loaded['X']
Y=loaded['Y']

# config parameters
dims = X.shape[1] # 2048 or size of Autoencoded vector (default 256)

# build APPROXIMATE NEAREST NEIGHBOR (ANN) index
t = AnnoyIndex(dims)
for index in tqdm(range(Y.shape[0])):
  file_vector = X[index]
  t.add_item(index, file_vector)
t.build(NUM_TREES)
# t.save(ANN_TREE)

def show_images(main_image, images, rows = 2):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of image paths.    
    rows (Default = 2): Number of rows in figure (number of cols is 
                        set to np.ceil(n_images/float(rows))).
    
    """
    n_images = len(images)
    fig = plt.figure()
    a = fig.add_subplot(rows, np.ceil(n_images/float(rows-1)), 1)
    img = mpimg.imread(main_image)
    plt.imshow(img)
    plt.axis('off')
    a.set_title("Target Image")    
    for n, image in enumerate(images):
        a = fig.add_subplot(rows, np.ceil(n_images/float(rows-1)), n + np.ceil(n_images/float(rows-1))+1)
        img = mpimg.imread(image)
        plt.imshow(img)
        plt.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def find_similar_images(target_image):
    """Find a list of similar images to a given image. 
    Parameters
    ---------
    target_image: Path of the target Image  
    """
    # build ann index
    # t = AnnoyIndex(dims)
    # t.load(ANN_TREE) # super fast, will just mmap the file
    x = img2vec.get_img_vector_single_img(target_image)
    if AUTO_ENCODER_ENABLE:
      from keras.models import load_model
      scaler = joblib.load(MIN_MAX_PICKLE) 
      encoder = load_model(AUTO_ENCODER_FILE)
      x = x.reshape(1,-1)
      x = scaler.transform(x)
      x = encoder.predict(x)
      x = x.flatten()
    master_vector = x
    named_nearest_neighbors = []
    nearest_neighbors = t.get_nns_by_vector(master_vector, NUM_SIMILAR_IMAGES)
    imgs =[]
    for j in nearest_neighbors:
      img = DATASET_PATH+Y[j].decode('UTF-8')
      imgs.append(img)
    return imgs

if __name__ == "__main__":
      imgfiles = glob.glob(TEST_IMAGES_FOLER+'*.jpg')
      for image in imgfiles:
          similar=find_similar_images(image)  
          show_images(image, similar)