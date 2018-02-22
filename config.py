# Enable AUTOENCODER
AUTO_ENCODER_ENABLE = False

# AUTOENCODER MODEL
AUTO_ENCODER_FILE = "rsc_files/autoencoder.h5"

# MIN MAX SCALER Object for Autoencoder vector
MIN_MAX_PICKLE = 'rsc_files/minmax.pkl'

# AUTOENCODED Image Vectors File (corresponding to Image Vectors order)
AUTO_ENCODED_VECTORS = 'rsc_files/x_ae.npz'

# Path to Dataset
DATASET_PATH = "mirflickr/"

# Image Vectors File
IMAGE_VECTORS = "rsc_files/image_vectors.npz"

# # ANN Tree
# ANN_TREE = "rsc_files/ann_tree.ann"

# Test Image Folder
TEST_IMAGES_FOLER = "test_images/"

# Uploads directory
UPLOAD_FOLDER = 'uploads/'          

# Number of nearest neighbors
NUM_SIMILAR_IMAGES = 5 

# Number of trees to build for ANN
NUM_TREES = 101                     
