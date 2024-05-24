# Automatic-Object-Labeling-Method-in-Panoramic-Images:

## Project Overview

Our method for automatically labeling objects in panoramic images leverages advanced computer vision techniques to classify objects in urban and road environments. By minimizing human intervention, we streamline the image analysis process, making it more efficient and scalable. This project is built using PyTorch, a powerful open-source machine learning library, enabling us to implement state-of-the-art models and achieve high accuracy in object detection and classification tasks.

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/bd8685db-3a83-4338-94d6-e9882fe6ca88)
![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/0af60eb7-4c78-4b57-9795-a8aff55ffb78)

This photograph used in the readme of this project comes from [SpyroSoft](https://spyro-soft.com/).

## Main Features

- **Automated Labeling:** Significantly reduces the need for manual intervention by automating the labeling of objects in panoramic images, enhancing productivity and consistency.
  
- **State-of-the-Art Computer Vision:** Implements advanced techniques such as transfer learning and fine-tuning to achieve high accuracy in object classification.
  
- **High Efficiency and Precision:** Focuses on optimizing both efficiency and precision in detecting and identifying objects within urban and traffic settings, ensuring reliable performance.
  
- **Robust Training Framework:** Utilizes renowned models like YOLOv8 and Fast R-CNN, leveraging their strengths for comprehensive training and fine-tuning of the detection algorithms.

## Project Block Diagram

Certainly! Here is a project block diagram for an automated object labeling system using YOLOv8 and Fast R-CNN with PyTorch:

![Diagrama de Bloques PF (2)](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/7a9c882e-7ba8-4d84-b6f4-b5f2086a92f6)

## Method Results with YOLOv8

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/f961108b-3044-426c-b866-447e5a11034f)
![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/11faeeff-f0e2-4ea7-a2f4-c212ed282089)

Here you can add a description of the results obtained using YOLOv8.

When looking at the results of the model retrainings we noticed that there was a noticeable improvement when adding the labeled images to the datasets and increasing their size and the quality of the retrainings, and consequently the quality of the inferences improved, i.e. the labeling method automatically improved more and more. This can be seen in the TrainModel-Inferences folders.

## Method Results with Fast R-CNN

![prediction_-_17MSu3kb40j3W1_2UU9A](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/a6980a3f-483c-430f-8ffe-f4e5ff645971)
![prediction_0A4LsIvFoqbtrfF7vgt1hA](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/78235737-c8ee-4c1b-bd26-01ae59bd4437)

Once the automatic object labeling method is implemented and trained using Fast R-CNN, it is imperative to conduct extensive testing to evaluate its performance and accuracy. These tests are classified into unit tests and end-to-end tests, allowing evaluation at both component and system levels.

Unit testing involves shorter training cycles and small data sets, allowing for meticulous analysis of how the model interprets and processes the data. This scrutiny is essential to perfect the code and understand the results obtained. Instead, comprehensive tests are performed on larger data sets and across multiple training epochs. The goal is to subject the model to various scenarios, ranging from ideal conditions to challenging environments, to measure its proficiency in detecting and labeling various objects in urban and road environments.

Throughout these tests, it is common to find initial uncertainty in the inference process of the Fast R-CNN model. This uncertainty often arises from overfitting, where the model fits closely to the training data, hindering its ability to generalize effectively to new instances. To mitigate this, strategies such as model retraining, hyperparameter optimization, and improving the quality and quantity of training data are essential. These measures facilitate performance optimization and improve the model's ability to generalize.

Although experimental, the combination of Fast R-CNN with reinforcement learning techniques has shown remarkable effectiveness in quickly generating predictions and labeling objects. As data collection techniques and algorithmic advances advance, the performance of these models is expected to see significant improvements. Continued exploration of novel techniques and methodologies is imperative to drive even further advances in automatic labeling of objects in images. Let us not forget to emphasize that this is a complete design but in certain parts it is experimental and as technologies advance we will be able to improve the implementation of data for more adverse situations.
## Project Structure

The project is structured as follows:

- **Dataset:** Contains a collection of pre-labeled images intended for retraining models. It also allows for the input of updated data, enabling continuous improvement and increased accuracy of the models over time.
- **Data:** This folder is the designated location for adding the panoramic images to be used in the project.
- **TrainModels-Inferences:** Includes the implementation of two distinct machine learning methods for labeling panoramic images: YOLOv8 and Fast R-CNN.
- **requirements_fast.txt:** Lists the dependencies required for running the Fast R-CNN code.
- **requirements_yolo.txt:** Lists the dependencies required for running the YOLOv8 code.

## Installation and Use

1. Clone this repository:
git clone https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images.git

3. Install the project dependencies:
pip install -r requirements_fast/yolo.txt

4. Run the code:
python (Fast/Yolo).py

It depends on the model you want to specialize in.

6. Review the generated labels and if necessary, re-execute the code modifying the hyperparameters

## Contribution
If you want to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: git checkout -b new-feature.
3. Make your changes and commit: git commit -m "Add new functionality."
4. Push to the branch: git push origin new-feature.
5. Send a pull request.

## Image Credits

- Photos taken from: [SpyroSoft](https://spyro-soft.com/)
- Database taken from: [Mappillary]([https://spyro-soft.com/](https://www.mapillary.com/dataset/places))

## Credits
This project was developed by Robinson Luis Campo Charris, Matthieu Navarro Chamucero and Luis Alejandro Vallejo Morales.

**Adviser:**

PhD. Juan Carlos Velez Diaz
