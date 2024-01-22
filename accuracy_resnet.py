import tensorflow
from tensorflow.keras.models import Model, load_model
import numpy as np
import pandas as pd
import os

SOURCE_DIR = 'data_area'
IMAGE_FOLDERS = [
    'crop2'
]
IMG_HEIGHT = 224
IMG_WIDTH = 224
VAL_DATA_FILE = 'data_area/val_data.csv'


def my_accuracy_function(model1):
    df_val = pd.read_csv(VAL_DATA_FILE)
    items_num_val = len(df_val['names']) * len(IMAGE_FOLDERS)
    y_test = np.zeros((items_num_val, 1))
    predict_test = np.zeros((items_num_val, 1))
    val_counter = 0

    score = 0
    incorrect = []

    for index, row in df_val.iterrows():
        name = row['names']
        value = row['value']
        for folder in IMAGE_FOLDERS:
            img_path = os.path.join(SOURCE_DIR, folder, name)
            print(img_path)
            img = tensorflow.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tensorflow.keras.utils.img_to_array(img)
            if value < 0.5:
                y_test[val_counter] = 0.
            else:
                y_test[val_counter] = 1.

            expanded_array = np.expand_dims(img_array, axis=0)
            pred = model1.predict(expanded_array)
            print("expected: " + str(y_test[val_counter]))
            print("predict: " + str(pred[0]))
            if pred[0] < 0.5:
                predict_test[val_counter] = 0.
            else:
                predict_test[val_counter] = 1.

            if predict_test[val_counter] == y_test[val_counter]:
                score = score + 1
            else:
                incorrect.append(name)
            val_counter = val_counter + 1

    print(incorrect)
    return float(score) / float(items_num_val)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    model = load_model('models/conv_next/04/model_best.06-0.0590.keras')
    # print(model.summary())

    result = my_accuracy_function(model)
    print(result)
