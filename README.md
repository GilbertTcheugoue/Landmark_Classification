# Landmark_Classification

# Project Overview

This project aims to  build models to automatically predict the location of the image based on any landmarks depicted in the image. Then, go through the machine learning design process end-to-end: performing data preprocessing, designing and training CNNs, comparing the accuracy of different CNNs, and deploying an app based on the best CNN you trained.

# Project Steps

1. Create a CNN to Classify Landmarks (from Scratch) - Here, We visualized the dataset, processed it for training, and then built a convolutional neural network from scratch to classify the landmarks. Then we  described some of our decisions around data processing and how we chose our network architecture. We then exported our best network using Torch Script.
   
2. Create a CNN to Classify Landmarks (using Transfer Learning) - Next, We investigated different pre-trained models and decided on one to use for this classification task. Along with training and testing this transfer-learned network, we explained how we arrived at the pre-trained network we chose. We also export our best transfer learning solution using Torch Script
   
3. Deploy your algorithm in an app - Finally, We used our best model to create a simple app for others to be able to use our model to find the most likely landmarks depicted in an image. We also testsd out our model  and reflected on the strengths and weaknesses of our model.


# Environment and Dependencies   

1. Clone the repository
   
2. Open a terminal and navigate to the directory of the repository.
   
3; Download and install Miniconda

4. Create a new conda environment with Python 3.7.6:
   
conda create --name environment_name python=3.7.6 pytorch=1.11.0 torchvision torchaudio cudatoolkit -c pytorch

Activate the environment:

conda activate environment_name


5. Install the required packages for the project:

pip install -r requirements.txt

6. Install and open jupyter lab:

pip install jupyterlab 

jupyter lab
