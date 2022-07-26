# Image Classification using AWS SageMaker

The project oriented  to use AWS Sagemaker for making a pretrained model that can perform image classification on provided dog-breed dataset by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.
This assignment is a part of AWS Machine Learning Engineer Nanodegree Program.

The are following tasks are done:

    - Using as a model pretrained Resnet34 net from pytorch vision library is used in the project
    - Fine-tuning the model with hyperparameter tuning and Network Re-shaping
    - Implement Profiling and Debugging with hooks
    - Deploy the model and perform inference

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found here (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

### Repo structure and contents
    - hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
    - train_model.py - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning
    - endpoint.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.
    - train_and_deploy.ipynb - This jupyter notebook contains all the code and the steps performed in this project and their outputs.

## Hyperparameter Tuning

The model is built with convolutional resnet34 pretrained model based on 34 layers. 
I have used for fine-tuning following hyperparameter ranges: 
1. "lr": ContinuousParameter(0.001, 0.1),
2. "batch_size": CategoricalParameter([32, 64, 128, 256])

Below is list of training jobs in the screenshot:
![Alt text](training_jobs.png?raw=true "training_jobs.png")
 
Hyperparameter tuning jobs completed jobs:
![Alt text](hpo_jobs.png?raw=true "hpo_jobs.png")
 
Logs from the last completed training job with metrics during the process:

 ![Alt text](logs.png?raw=true "logs.png")


## Debugging and Profiling

During of training I didn’t know model performance, whether accuracy is acceptable. For visualization was applied graphical representation of the Cross Entropy Loss which is shown below.
![Alt text](curves.png?raw=true "curves.png")
 

### Results
I have tried a few training jobs, here is summary with metrics and insights of last training job: 
From shown information about trained job, built model is not overfitted, some errors are in overtraining, so need more investigate to weight initialization, probably running on GPU may impact(trigger error, so was used cpu)
## Model Deployment

The model was deployed in ml.m5.xlarge instance and used cuda-cpu as device for training.
There are deployed endpoints in inferences.
 
For querying the endpoint is needed to add the path of test data image by opening for reading this file and triggering predict function of model.


