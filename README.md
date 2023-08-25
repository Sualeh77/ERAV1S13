Object Detection using YoloV3 in Pytorch Lightning

### Summary

This project is an Object Detection project. Dataset used for training is PascalVOC. Model used for traning is YoloV3 with DarkNet implemented. Albumentation is used for image transformation. One Cycle LR policy is used for best LR range.

Model is trained for 40 epochs, with batch size of 32. For Identifying reasons of incorrect predictions Grad-CAM is implemented.

File Structure

### S13_Colab.ipynb :
Training Is done in this notebook. 
 In this notebook all of the defined implementations in project are fetched and used. EG: Testing train/test data loader, Visualising sample images and effect of transformations, Use of LR finder to find max best LR, Running Training pipeline, Visualising Grad-CAM heatmap etc...

### models :

This is the directory where models are stored. It currently includes a pytorch model and a Lightning model with pytorch model as it's base.

### dataset.py :

Dataset class is definrd in this.

### datamodule.py

in this file lightning data hooks are defined.


### utils.py :

Utitlity functions are kept in this file