# text_recognition

This project is the implementation of "End-to-End Text Recognition with Convolutional Neural Networks" (Wang et al.) http://ai.stanford.edu/~ang/papers/ICPR12-TextRecognitionConvNeuralNets.pdf, using tensorflow and opencv on python.

The goal of this project is to detect and recognize texts or letters from an image. In practical situations, furthermore, it can be combined with other technologies such as translation to provide useful information to users. 


## data sets

For training neural networks for recognition and detection, we are going to use ICDAR 2003 dataset, which contains 6185 images of numbers and characters http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/char.zip, and University of Surreys' 74k images of numbers and characters http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz. We convert all of the images to grayscale and resize to 32x32 so that the shape of input layers of the networks is 32x32x1. A label is 62 dimentional vector with all zeros but one 1.0 whose index is equal to the class of the image. We combine them to one 2d ndarray with shape (number of datasets, 1086). The constructed ndarray is saved in 'data' file under each directory contains datasets. Non character 32x32 image patches for training CNN for detection are generated from ICDAR 2003 dataset(character centered patches are removed by human hands), and dataset is constructed with the same structure. 

## convolutional neural networks

We use CNN to detect and recognize text from an image. As the paper suggests, the two networks have an identical structure except the number of filters of convolutional layers, n1 and n2. 

32x32x1 (input layer)->
25x25xn1 (convolutional layer1)->
5x5xn1 (average pooling layer1)->
4x4xn2 (convolutional layer2)->
2x2xn2 (average pooling layer2)->
 1024 (fully connected layer with dropout)->
2(detection) or 62(recognition) (fully connected layer to output).

As you can see, we make some minor changes from the original paper such as the first fully connected layer with 1024 neurons. 
