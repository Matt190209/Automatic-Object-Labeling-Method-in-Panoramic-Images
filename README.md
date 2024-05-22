# Automatic-Object-Labeling-Method-in-Panoramic-Images:

## Project Overview

Our method for automatically labeling objects in panoramic images leverages advanced computer vision techniques to classify objects in urban and road environments. By minimizing human intervention, we streamline the image analysis process, making it more efficient and scalable. This project is built using PyTorch, a powerful open-source machine learning library, enabling us to implement state-of-the-art models and achieve high accuracy in object detection and classification tasks.

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/bd8685db-3a83-4338-94d6-e9882fe6ca88)
![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/0af60eb7-4c78-4b57-9795-a8aff55ffb78)

Photos taken from: https://spyro-soft.com/

## Main Features

- **Automated Labeling:** Significantly reduces the need for manual intervention by automating the labeling of objects in panoramic images, enhancing productivity and consistency.
  
- **State-of-the-Art Computer Vision:** Implements advanced techniques such as transfer learning and fine-tuning to achieve high accuracy in object classification.
  
- **High Efficiency and Precision:** Focuses on optimizing both efficiency and precision in detecting and identifying objects within urban and traffic settings, ensuring reliable performance.
  
- **Robust Training Framework:** Utilizes renowned models like YOLOv8 and Fast R-CNN, leveraging their strengths for comprehensive training and fine-tuning of the detection algorithms.

## Project Block Diagram

Certainly! Here is a project block diagram for an automated object labeling system using YOLOv8 and Fast R-CNN with PyTorch:

![Diagrama de Bloques PF (2)](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/7a9c882e-7ba8-4d84-b6f4-b5f2086a92f6)

## Method Results with YOLOv8

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/4814aea8-66b6-4799-82e9-4b2265a9b281)
![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/bafee9dc-867a-43da-9166-ee476b4eff89)

Here you can add a description of the results obtained using YOLOv8.

## Method Results with Fast R-CNN

![Captura de pantalla Fast R-CNN](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/892a85c8-7401-432e-b507-a6997018904b)

Here you can add a description of the results obtained using Fast R-CNN.

## Project Structure

The project is structured as follows:

- **Dataset:** Contains a dataset of previously labeled images, designed for retraining already pretrained models. Additionally, it provides the ability to enter revised data, allowing you to continue improving model learning and increasing its accuracy over time.
- **Data:** It is the destination folder where you should add the panoramic images to use
- **TrainModels:** Contains the two different Machine Learning methods for labeling panoramic images.
- **Requirements_fast.txt:** Requirements file with the dependencies of the Fast R-CNN code
- **Requirements_yolo.txt:** Requirements file with the dependencies of the Yolo_V8 code

## Installation and Use

1. Clone this repository:
git clone https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images.git

3. Install the project dependencies:
pip install -r requirements_fast/yolo.txt

4. Run the code:
python algo.py

5. Review the generated labels and if necessary, re-execute the code modifying the hyperparameters

## Contribution
If you want to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: git checkout -b new-feature.
3. Make your changes and commit: git commit -m "Add new functionality."
4. Push to the branch: git push origin new-feature.
5. Send a pull request.

## Créditos
Este proyecto fue desarrollado por Robinson Luis Campo Charris, Matthieu Navarro Chamucero y Luis Alejandro Vallejo Morales.

 



