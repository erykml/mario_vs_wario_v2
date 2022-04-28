# script for training a CNN classifier 

from config import PROCESSED_IMAGES_DIR, MODELS_DIR
from scrt import *
import os
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
from dagshub import dagshub_logger

mlflow.set_tracking_uri("https://dagshub.com/eryk.lewinson/mario_vs_wario_v2.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = PASSWORD

def get_datasets(validation_ratio=0.2, target_img_size=64, batch_size=32):

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       zoom_range=[0.5, 1.5],
                                       validation_split=validation_ratio)

    valid_datagen = ImageDataGenerator(rescale=1./255, 
                                       validation_split=validation_ratio)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(f"{PROCESSED_IMAGES_DIR}/train",
                                                     target_size = (target_img_size, target_img_size),
                                                     color_mode="grayscale",
                                                     batch_size = batch_size,
                                                     class_mode = "binary",
                                                     shuffle=True,
                                                     subset="training")

    valid_set = valid_datagen.flow_from_directory(f"{PROCESSED_IMAGES_DIR}/train",
                                                  target_size = (target_img_size, target_img_size),
                                                  color_mode="grayscale",
                                                  batch_size = batch_size,
                                                  class_mode = "binary",
                                                  shuffle=False,
                                                  subset="validation")

    test_set = test_datagen.flow_from_directory(f"{PROCESSED_IMAGES_DIR}/test",
                                                target_size = (target_img_size, target_img_size),
                                                color_mode="grayscale",
                                                batch_size = batch_size,
                                                class_mode = "binary")

    return training_set, valid_set, test_set

if __name__ == "__main__":

    
    mlflow.tensorflow.autolog()

    IMG_SIZE = 64
    LR = 0.01

    with mlflow.start_run():
        training_set, valid_set, test_set = get_datasets(validation_ratio=0.2, 
                                                         target_img_size=IMG_SIZE, 
                                                         batch_size=32)

        # Initialising 
        model = Sequential()

        # 1st conv. layer
        model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 1), activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # 2nd conv. layer
        model.add(Conv2D(32, (3, 3), activation = "relu")) #no need to specify the input shape
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # 3nd conv. layer
        model.add(Conv2D(64, (3, 3), activation = "relu")) #no need to specify the input shape
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # Flattening
        model.add(Flatten())

        # Full connection
        model.add(Dense(units = 64, activation = "relu"))
        model.add(Dropout(0.5)) # quite aggresive dropout, maybe reduce
        model.add(Dense(units = 1, activation = "sigmoid"))

        model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate=LR),
                      loss = "binary_crossentropy", 
                      metrics = ["accuracy"])

        model.fit(training_set,
                  validation_data=valid_set,
                  epochs = 10)

        test_loss, test_accuracy = model.evaluate(test_set)

        with dagshub_logger() as logger:
            logger.log_metrics(loss=test_loss, accuracy=test_accuracy)
            logger.log_hyperparams({
                "img_size": IMG_SIZE,
                "learning_rate": LR
            })

        mlflow.log_params({
            "img_size": IMG_SIZE,
            "learning_rate": LR
        })
        mlflow.log_metrics(
            {
                "test_set_loss": test_loss,
                "test_set_accuracy": test_accuracy,
            }
        )

        model.save(MODELS_DIR)