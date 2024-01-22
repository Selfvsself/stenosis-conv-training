import tensorflow
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

SOURCE_DIR = 'data_area'
IMAGE_FOLDERS = [
    'crop2'
]
TRAIN_DATA_FILE = 'data_area/train_data.csv'
VAL_DATA_FILE = 'data_area/val_data.csv'
NUM_CLASSES = 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16


def train():
    df_train = pd.read_csv(TRAIN_DATA_FILE)
    items_num_train = len(df_train['names']) * len(IMAGE_FOLDERS)
    x_train = np.zeros((items_num_train, IMG_HEIGHT, IMG_WIDTH, 3))
    y_train = np.zeros((items_num_train, NUM_CLASSES))
    train_counter = 0

    for index, row in df_train.iterrows():
        name = row['names']
        value = row['value']
        for folder in IMAGE_FOLDERS:
            img_path = os.path.join(SOURCE_DIR, folder, name)
            print(img_path)
            img = tensorflow.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tensorflow.keras.utils.img_to_array(img)
            x_train[train_counter] = img_array
            y_train[train_counter] = value
            train_counter = train_counter + 1

    df_val = pd.read_csv(VAL_DATA_FILE)
    items_num_val = len(df_val['names']) * len(IMAGE_FOLDERS)
    x_test = np.zeros((items_num_val, IMG_HEIGHT, IMG_WIDTH, 3))
    y_test = np.zeros((items_num_val, NUM_CLASSES))
    val_counter = 0

    for index, row in df_val.iterrows():
        name = row['names']
        value = row['value']
        for folder in IMAGE_FOLDERS:
            img_path = os.path.join(SOURCE_DIR, folder, name)
            print(img_path)
            img = tensorflow.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tensorflow.keras.utils.img_to_array(img)
            x_test[val_counter] = img_array
            y_test[val_counter] = value
            val_counter = val_counter + 1

    conv_base = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_tensor=None,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )

    conv_base.trainable = True

    # print(conv_base.summary())

    data_augmentation = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.RandomFlip("horizontal"),
            tensorflow.keras.layers.RandomRotation(0.2),
            tensorflow.keras.layers.RandomZoom(0.1),
            tensorflow.keras.layers.RandomContrast(0.1),
            tensorflow.keras.layers.RandomBrightness(0.1),
        ]
    )
    inputs = tensorflow.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(scale=1. / 255)(x)
    x = conv_base(x)
    x = tensorflow.keras.layers.Flatten()(x)
    output = tensorflow.keras.layers.Dense(64, activation='relu')(x)
    output = tensorflow.keras.layers.Dropout(0.2)(output)
    output = tensorflow.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(output)
    model = tensorflow.keras.Model(inputs, output)
    #
    print(model.summary())
    #
    # # model = load_model('models/resnet_v2/01/model_last.keras')
    #
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0002),
        loss='mse',
        metrics=["accuracy"])
    #
    dir_path = "models/resnet_v2/01/"
    os.makedirs(dir_path)
    file_name = "model_best.{epoch:02d}-{loss:.4f}.keras"
    filepath = dir_path + file_name

    earlystopper = EarlyStopping(patience=20, verbose=1)

    checkpoint = ModelCheckpoint(filepath, period=1)

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=100,
                        shuffle=True,
                        callbacks=callbacks_list)

    model.save(dir_path + 'model_last.keras')

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(dir_path + 'out.csv', index=False)

    plt.figure(figsize=(15, 15))
    plt.plot(history.history['loss'],
             label='Показатель ошибок на обучающем наборе')
    plt.plot(history.history['val_loss'],
             label='Показатель ошибок на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Показатель ошибок')
    plt.legend()
    plt.savefig(dir_path + 'loss.png')

    plt.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'],
             label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.savefig(dir_path + 'acc.png')


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    train()
