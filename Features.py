from pickle import dump, load
import os 
from keras.applications.xception import Xception, preprocess_input
from tqdm.auto import tqdm
import numpy as np
directory = '\Flickr8k_Dataset'
def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for (imgs) in tqdm(os.listdir(directory)):
            filename = directory + "/" + imgs
            image =  image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0

            feature = model.predict(image)
            features[imgs] = feature
        return features
dataset_images = "D:\Python\Flickr8k_Dataset"
#2048 feature vector
features = extract_features(dataset_images)
dump(features, open("features.p","wb"))
