# ACT-Traffic-Accident Model

## Datasets ##
This is the data set I used to train the model.
https://universe.roboflow.com/mada-study/crash-car-detection

Provided by a Roboflow user
License: CC BY 4.0

## Setup ##
Project tested using Python 3.8.5  
Install the following pip/conda packages:
* dlib
* cmake

Install the requirements.txt from this repo:  
```pip install -r requirements.txt```

If you are using Anaconda and there's a complaint about "version `GLIBCXX_3.4.29' not found", then run  
```conda install -c conda-forge gxx_linux-64==11.1.0```

## Running detections ##

To run the detecttion model do the following:

1. Place the video or images that you want to run the model on in the data folder. I have already included some test videos in the data/video folder.
2. You will then choose the weight for the model which is bestCrash.pt in this case.
3. Finally run the following in the command prompt: (run it with "data/video/(name of the video here)")


```python detect.py --weights bestCrash.pt --img 640 --conf 0.25 –-source data/video/(Name of Video Here)```

The results of running the model will appear in runs/detect/

