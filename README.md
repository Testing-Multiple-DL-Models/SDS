# Measuring Discrimination to Boost Comparative Testing for Multiple Deep Learning Models
This repository stores our experimental codes for the paper "Measuring Discrimination to Boost Comparative Testing for Multiple Deep Learning Models". SDS is short for the approach we proposed in this paper: Sample Discrimination based Selection.
## Datasets
The datasets we used (MNIST,Fashion-MNIST,CIFAR-10) all can be loaded through python's keras package, such as:  
```python
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```
## Main experimental codes
You can easily reproduce our method or modify the code we provide.
  
We have provided the code of our method and uploaded 25 models related to the Fashion-MNIST dataset. 
  
Our method is called SDS for short, and you can find the code of our method in the folder named code, which is named SDS.py. And in the ‘models’ folder, you can find 25 models we used for Fashion-MNIST. You can run SDS.py directly to reproduce our original experimental results. If you want to conduct your own experiments, you can modify the model and code accordingly.  
  
You should create corresponding folders to store the results of different models，i.e., the accuracy for each model.
## Baseline methods
We have used four baseline methods in total, we will explain the experimental settings of the baselines.
  
**'SRS'**: short for the Simple Random Selection.  
  
**'CES'**: short for the method proposed in the paper 'Boosting Operational DNN Testing Efficiency through Conditioning'. CES is a sampling method for a single model. Since our experiment is for a multi-model scenario, we use the CES method to sample for all models, and then select the best performing subset as the baseline. The best performance here means that we use the sampled subset of the model to measure the spearman coefficients and jaccard coefficients of all model accuracy (35-180) and the final model accuracy, and calculate the mean value of all points, and the largest mean value is regarded as the optimal subset.   
  
**'RDG'**: RDG is the baseline we used after randomization according to the method proposed in the paper 'DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks'. For each sample, we calculated the maximum/minimum ξ values of all models, and selected samples according to sampling size, randomly sampled in the first 25%. We reported the best of them in the paper. 
  
**'DDG'**: This is also using the method in the paper 'DeepGini: Prioritizing Massive Tests to Enhance the Robustness of Deep Neural Networks'. But this time we do not perform random sampling, but take the top 180 marks ranked as the test subset. In other words, this is a deterministic sampling method.  
  
