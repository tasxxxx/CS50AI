I first started out with the CNN code given in the notes. This would serve as my base which then I will make changes to in order to improve the accuracy. These were the changes I had made and what I had noticed:

1. I decided to include more more convolutional and pooling layers. This was because convolution and pooling layers are used to abstract important features from the images. The more features that the CNN learns, the better it will predict. Moreover, I increased the number of filters with each subsequent layer. This was so that the CNN starts off with learning the 'big and simple' features and then moves on to learn the 'finer and complex' details.

2.  I increased the kernel filter size so that the CNN 'sees' more features.

3. I decided to not change the activation function for all the layers. I stuck with relu as this ensures that subsequent neurons are activated only when its input is more than zero. I also stuck with softmax for the final output as this calculates the probability of the image falling under a category.

4. I left the pool size unchanged as changing the size results in an error.

5. I did not add any extra hidden layers as doing so resulted in the accuracy to decrease. This is probably due to the model overfitting on the training set.