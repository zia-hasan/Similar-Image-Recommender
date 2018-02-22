# """
# An autoencoder is an artificial neural network used for unsupervised learning of efficient codings. 
# An autoencoder can learn a representation for a set of data, typically for the purpose of dimensionality 
# reduction.If linear activations are used, or only a single sigmoid hidden layer, then the optimal solution
# to an autoencoder is strongly related to principal component analysis (PCA). With appropriate dimensionality
# and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or 
# other basic techniques. It turns out that PCA only allows linear transformation of a data vectors. 
# Autoencoders and RBMs, on other hand, are non-linear by the nature, and thus, they can learn more 
# complicated relations between visible and hidden units. Moreover, they can be stacked, which makes them even more powerful.
# """
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import sys
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from tqdm import tqdm
from config import *
from sklearn.externals import joblib 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.dont_write_bytecode=True

dims = 2048 # Original Dimension from Inceptionnet
encoding_dim = 256 # Encoding Dimension

loaded = np.load(IMAGE_VECTORS)
X=loaded['X']
Y=loaded['Y']

print(X.shape)
print(Y.shape)

# Scale each feature in (0,1) range 
scaler = MinMaxScaler()
sX = scaler.fit_transform(X)

# We use all data now (0.95) because this model was already tested 
X_train, X_test, Y_train, Y_test = train_test_split(sX, Y, train_size = 0.95, random_state = np.random.seed(2017))

### DEEP AUTOENCODER WITH MULTIPLE LAYERS ###
input_layer = Input(shape = (dims, ))

# Encoder Layers
encoded1 = Dense(1024, activation = 'relu')(input_layer)
encoded2 = Dense(512, activation = 'relu')(encoded1)
encoded3 = Dense(256, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)
# Decoder Layers
decoded1 = Dense(256, activation = 'relu')(encoded4)
decoded2 = Dense(512, activation = 'relu')(decoded1)
decoded3 = Dense(1024, activation = 'relu')(decoded2)
decoded4 = Dense(dims, activation = 'sigmoid')(decoded3)

# Combine Encoder and Decoder into an Autoencoder Model
autoencoder = Model(inputs = input_layer, outputs = decoded4)
# Configure and Train the Autoencoder with mean squared error
autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')
# Fit the model
autoencoder.fit(X_train, X_train, nb_epoch = 30, batch_size = 256, shuffle = True, validation_data = (X_test, X_test))

# Extract the Encoder part from the Autoencoder Model for dimensionality reduction
encoder = Model(inputs = input_layer, outputs = encoded4)

# Obtain Auto-encoded vectors
X_ae = encoder.predict(sX)

# Save Files
encoder.save(AUTO_ENCODER_FILE)
np.savez_compressed(AUTO_ENCODED_VECTORS, X=X_ae)
joblib.dump(scaler, MIN_MAX_PICKLE) 