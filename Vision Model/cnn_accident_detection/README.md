# ACT CV Crash Detection

## Datasets
https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage

## Setup
Project tested using Python 3.10.7

Install the requirements.txt
```pip install -r requirements.txt```

## Running the Application
```python accident_detection.py <path_to_video_file>```

Video file should be in a format like `.mkv` or `.mp4`.

To terminate the application while running, press "s" key.

Application requires `model.json` and `model_weights.h5` file to be generated using one of the two provided
classification notebooks that can be run with Jupyter Notebook (`classification_cnn1.ipynb` or `classification_cnn2.ipynb`).

At the time of this README, `classification_cnn2` provides a higher prediction accuracy for the test set with 50 epochs:
```
Epoch 50/50
8/8 [==============================] - ETA: 0s - loss: 0.0504 - accuracy: 0.9810
Epoch 50: val_accuracy did not improve from 0.91837
8/8 [==============================] - 11s 1s/step - loss: 0.0504 - accuracy: 0.9810 - val_loss: 0.4561 - val_accuracy: 0.8673
```


The CNN weight classifier notebooks were adapted/modified from the Kaggle example here: https://www.kaggle.com/code/fahaddalwai/cnn-accident-detection-91-accuracy




