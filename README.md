# Digit-Recognition-Google-Tensorflow

### Overview

Used Google Tensorflow to build a neural network on MNIST dataset (60000 training images and 10000 test images) and attained a modest accuracy of 96.26%. For memory computation, I used memory-profiler / resource and for time computation, I used both time() and tensorflow’s own timeline library to obtain time resource graph. I also explored tfprof library and think this would be the best option for a comprehensive profiling of both time and memory especially on large datasets.

### About the Neural Network:

Three hidden layers each with 400, 400, and 500 nodes respectively
Trained the data in a batch size of 100
No of epochs = 20
Activation function used is Rectified Linear Unit (ReLU)
Cost computation -> softmax_cross_entropy_with_logits
Optimizer -> AdamOptimizer algorithm

### Memory Usage on Local Machine ~ 12 MB

### Estimating time and resources required to handle a training set of 60 Billion images and a test set of 10 Billion

Assessing the performance of this model (both time and resources) on the newly launched Google TPUs (Tensor Processing Units)

##### How fast is TPU compared to GPU? 

Following calculations are based on <a href="https://techcrunch.com/2017/05/17/google-announces-second-generation-of-tensor-processing-unit-chips/">this</a> article where a Google spokesperson said, “To put this into perspective, our new large-scale translation model takes a full day to train on 32 of the world’s best commercially available GPU’s—while one 1/8th of a TPU pod can do the job in an afternoon,” Google wrote in a statement.

Full Day = 24 hours * 32 GPUs = 768 hours / GPU
1/8TPU * 8 hours = 1 hour / TPU

1 hour / TPU = 768 hours / GPU.            ---------------------------    1

##### How fast is GPU compared to CPU?

Based on this <a href="https://blogs.nvidia.com/blog/2010/06/23/gpus-are-only-up-to-14-times-faster-than-cpus-says-intel/">blog</a> which is a conservative estimate of a GPU’s speed by Intel, 

1 hour / GPU = 14 hours / CPU.              ---------------------------    2


From 1 and 2, we can derive

1 hour / TPU = 10752 hours / CPU         -------------------------    3

Based on all of this information and our own results from the model implementation, it is safe to assume that running the same model on TPU will take following time:


 135 seconds * 1 Million / (10752 * 3600) = 3.49 seconds 

This is quite an amazing result that even after scaling up the data by a million, the same model can be implemented at a speed of ~3.5 seconds which is 1/38th times the speed of a normal CPU (in our case, a 2.3 GHz 64-bit processor (i5)). 

Further, <a href="https://tech.geekboots.com/">this</a> article suggests that one TPU can handle the processing of over 1M regular Google images per day (which have far more memory consumption due to enhanced quality, color, density etc. compared to the MNIST greyscale images). Further, a cloud TPU resource accelerates the performance of linear algebra computation, minimizes the time-to-accuracy training large, complex neural network models like in our case. Models that previously took weeks to train on other hardware platforms can converge in hours on TPUs. <a href="https://www.forbes.com/sites/moorinsights/2018/02/13/google-announces-expensive-cloud-tpu-availability/#5e495a31359f">This</a> table is a comparative snapshot which gives the pricing as well.


Recommendation

In conclusion, based on the research, I recommend using Google cloud TPUs for better performance and scalability in minimum time even though the cost is a bit on the higher side.



