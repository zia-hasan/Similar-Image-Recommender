# Image Similarity with Transfer Learning and AutoEncoder
---
## Introduction
This is an implementation of image similarity with tranfer learning from Inception net model with an optional Autoencoder. There are various classical methods for image classification such as DHASHING etc. but neural networks have recently caught up and can infer abstract information. There have been quite a few publications using transfer learning from state of the art models to extract pre-trained weights and associated techniques. I consider the penultimate layer of Inception net model as a vector representation of an input image (after forward propagating it through the model). The hypothesis is that the 2048 activations from penultimate layer provide a decent abstract information about the image and for any new input image we just need to search its nearest neighbors from the similar image vectors of the dataset. Image vectors of the dataset can be precomputed and stored in a dump file.

Since 2048 is still a large sized vector, scaling it to a large dataset to search nearest neighbors can be quite slow and tricky. Therefore, I propose we use an autoencoder on top to reduce the vector size to 256. The structure of the autoencoder can be found in the code. 

A reasonable size dataset is mirflickr25k. We need to download the dataset, extract it and remove unneccesary folders and keep just images:

* `wget http://press.liacs.nl/mirflickr/mirflickr25k.v2/mirflickr25k.zip`
* `unzip mirflickr25k.zip`
* `cd mirflickr`
* `rm -rf doc`
* `rm -rf meta`

---
## Project code
All main part of code for this project is implemented in the following files:

* Constants Used: `config.py`
* Code to extract penultimate layer activations of inceptionnet: `img2vec.py`
* Computing the nearest neighbor using annoy library: `ann.py`
* Webservice: `app.py`
* HTML templates folder: `templates/`
* Resource files such as autoencoder files, processed vectors: `rsc_files/`

## Requirements and Setup
We need tensorflow, keras, flask, numpy, matplotlib, annoy etc. We need to download inceptionet model weights(inception-2015-12-05.tgz) and tf associated graph by running `python img2vec.py`. This stores the inceptionnet weights in a temporary directory. To generate image vectors from the dataset we need to run `python img2vec.py "mirflickr/*"` (or wherever dataset lies)

The image similarity on images from a test folder (set up in config file)
can be found by simply running `python ann.py`. Optionally Autoencoder files can be generated with `python autoencoder.py`.

The web-service can be launched by `python app.py`

[//]: # (Image References)

[image1]: ./readme/1.png "Example1"
[image2]: ./readme/2.png "Example2"
[image3]: ./readme/3.png "Example3"
[image4]: ./readme/4.png "Example4"
[image5]: ./readme/5.png "Example5"
[image6]: ./readme/6.png "Example6"

---

### Discussion and Issues Faced
This project was challenging but since I had little background in image similarity etc and time was limited, neural network based approach using a pre-trained state of the art model like InceptionNet seemed like a fastest and most accurate approach. In addition an Autoencoder on the penultimate layer activations could theoretically reduce the search space. Using Autoencoder on the whole dataset of images would have taken a lot of time on training and would have required extra computing resources. The penultimate layer gives 2048 activations and it was initially slow to search nearest neighbors but after playing with some parameters, I found a good balance. Using the autoencoder suprisingly works well on most images but fails to detect similarity sometimes. One can try plain PCA instead of autoencoder as another suggested approach. We can play with the configuration file and autoencoder structure to try the autoencoder version. 

The nearest neighbors can be approximated by approximate nearest neighbors algorithms. There are multiple implementations available online. Spotify has a nice implementation called `annoy` python library and facebook created `faiss` library. Which library to use is a subject of discussion but I used `annoy` because it seemed fast and easier to implement. Choosing number of trees is one challenge. The higher, the better accuracy but slower. Around 100 seemed like a decent trade-off number.  

The current implementation isn't exactly optimized for web application but it is just proof of concept and with some clever tricks, one can write a more optimized version.

The biggest issue was my lack on knowledge on web-development and transforming it into a web service but after reading a few articles and watching soem youtube videos, I was able to create a simple Flask app to do the job 

### Samples
Here are some samples. Surprisingly, the application can detect quite abstract information but also finds uncorrelated images too sometimes
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]