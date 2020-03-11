# Image-Classifier-Project
A flower image classifier project built with Python. A pretained VGG model is used to implement the classification. It classifies that which flower category that the image belongs.

### Getting started
Install jupyternotebook and open the *Image Classifier Project.ipynb*. Train the model with this notebook. <br/>
When the model is ready, run in terminal to test the image:
```
python --input /path/to/filename.png --checkpoint /path/filename.pth --top_k 3 --category_name cat_to_name.json
```

### Installation
#### Pre-requisite
* Python 3.6
* Numpy
* Matplotlib
* PIL
* Pytorch
* Anaconda
* Jupyter notebook


### Acknowledgements
This project is part of Udacity Nanodegree Program that requires the students to complete in order to graducate from this program.