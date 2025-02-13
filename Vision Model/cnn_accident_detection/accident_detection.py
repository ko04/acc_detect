from keras.models import model_from_json
import cv2
import numpy as np
import sys

class AccidentDetectionModel(object):
    def __init__(self, model_json_file, model_weights_file):
        self.class_nums = ['Accident', 'Normal']

        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.predictions = self.loaded_model.predict(img)
        return self.class_nums[np.argmax(self.predictions)], self.predictions

def run(model, video_name):
    video = cv2.VideoCapture(video_name)
    while True:
        # read the next frame from the video
        _, frame = video.read()

        # convert the frame to grayscale and resize the region of interest
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # make a prediction using the model and update the frame if necessary
        prediction, probability = model.predict_accident(roi[np.newaxis, :, :])
        if prediction == 'Accident':
            probability_percent = round(probability[0][0] * 100, 2)

            # draw a black rectangle and put text on the frame
            cv2.rectangle(frame, (0, 0), (300, 50), (0, 0, 0), -1)
            text = f'{prediction} {probability_percent}%'
            cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

        # check if the 's' key is pressed and exit if it is
        if cv2.waitKey(99) & 0xFF == ord('s'):
            return

        # show the updated frame
        cv2.imshow('Video', frame) 


if __name__ == '__main__':
    run(AccidentDetectionModel('model.json', 'model_weights.h5'), sys.argv[1])