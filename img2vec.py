"""
This is a modification of the classify_images.py
script in Tensorflow. The original script produces
string labels for input images (e.g. you input a picture
of a cat and the script returns the string "cat"); This
modification reads in a directory of images and 
generates a vector representation of the image using
the penultimate layer of neural network weights. Also generates 
vector representation of a single image.

Usage: python classify_images.py "../image_dir/*.jpg" to generate vectors for a dataset

"""
from __future__ import absolute_import, division, print_function
import os.path
import os
import sys
import tarfile
import glob
import psutil
import numpy as np
import tensorflow as tf
from collections import defaultdict
from six.moves import urllib
from config import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.dont_write_bytecode=True

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def get_img_vector_single_img(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Image Vector representation
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()
  feature_vector = None

  with tf.Session() as sess:
    # Some useful tensors:
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.

    # Get penultimate layer weights
    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    feature_set = sess.run(feature_tensor,
                    {'DecodeJpeg/contents:0': image_data})
    feature_vector = np.squeeze(feature_set)

  # close the open file handlers
  proc = psutil.Process()
  open_files = proc.open_files()
  for open_file in open_files:
    file_handler = getattr(open_file, "fd")
    os.close(file_handler)

  return feature_vector        

def generate_img_vectors_from_images(image_list, output_dir):
  """Runs inference on an image list.

  Args:
    image_list: a list of images.
    output_dir: the directory in which image vectors will be saved

  Returns:
    Nothing
  """
  image_to_labels = defaultdict(list)

  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.

    x = []
    y = []

    for image_index, image in enumerate(image_list):
      try:
        print("parsing", image_index, image)
        if not tf.gfile.Exists(image):
          tf.logging.fatal('File does not exist %s', image)
        
        with tf.gfile.FastGFile(image, 'rb') as f:
          image_data =  f.read()

          # Get penultimate layer weights
          feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
          feature_set = sess.run(feature_tensor,
                          {'DecodeJpeg/contents:0': image_data})
          feature_vector = np.squeeze(feature_set)

          x.append(feature_vector)
          y.append(image)        
        # close the open file handlers
        proc = psutil.Process()
        open_files = proc.open_files()

        for open_file in open_files:
          file_handler = getattr(open_file, "fd")
          os.close(file_handler)
      except:
        print('could not process image index',image_index,'image', image)
    X = np.array(x)
    Y = np.array(y)
    np.savez_compressed(IMAGE_VECTORS, X=X, Y=Y)


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  print(FLAGS.model_dir)
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  maybe_download_and_extract()
  if len(sys.argv) < 2:
    print("please provide a glob path to one or more images, e.g.")
    print("python classify_image_modified.py '../cats/*.jpg'")
    sys.exit()

  else:
    output_dir = "image_vectors"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    images = glob.glob(sys.argv[1])
    image_to_labels = generate_img_vectors_from_images(images, output_dir)

    print("all done")
if __name__ == '__main__':
  tf.app.run()
