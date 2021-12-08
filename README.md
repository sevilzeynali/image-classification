# Image classification by transfer learning

It is a  Python program for image classification. Complete dataset is available [here](https://www.kaggle.com/ikarus777/best-artworks-of-all-time?select=images). I used a sample of 4 classes available in dataset.zip.

## Installing and requirements
You need to install :

 - Python
 - Pandas
 - Matplotlib
 - Sklearn
 - MLflow
  
## How does it work
To run this program you shoud do in a terminal or conda environment
```
paintings_classification.py
 ```
 for tracking the model with MLflow you can type this localhost in your browser:
 ```
 http://localhost:5000
 ```
## More information
This program uses EfficientNetB1 model to transfer learning. For evaluating this model we use confusion matrix.
 
