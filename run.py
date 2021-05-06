import librosa
import numpy as np
import keras
import pandas as pd
import sys
import os
import csv


# loading json and creating model
from keras.models import model_from_json

foldername = sys.argv[1]

files = os.listdir(foldername)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")


preds = {}
for filename in files:
    if filename.endswith('.wav'):
        X, sample_rate = librosa.load(f"{foldername}/{filename}", res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)

        livepreds = loaded_model.predict(twodim, 
                                batch_size=32, 
                                verbose=1)

        livepreds1=livepreds.argmax(axis=1)

        liveabc = livepreds1.astype(int).flatten()

        print(liveabc)
        

        preds[filename] = liveabc[0]

        print(f"\nFinished prediction on {filename}\n")

        #livepredictions = (lb.inverse_transform((liveabc)))

        #print(livepredictions)


print("Writing the results to predictions.csv!")
with open('predictions.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for key, value in preds.items():
        csv_writer.writerow([key, value])

print("Writing finished!")


