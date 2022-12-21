import argparse
import os
import warnings

import pandas as pd
import numpy as np
from PIL import Image
import sys
import logging
from sklearn.model_selection import train_test_split
import os, os.path
import json
import joblib

from sklearn.preprocessing import OneHotEncoder
import joblib
import logging
import copy
from sklearn.exceptions import DataConversionWarning


import tensorflow as tf
import os
import model
import joblib


warnings.filterwarnings(action="ignore", category=DataConversionWarning)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_test", type=float, default=0.3)
    parser.add_argument(
        "--input_path",
        type=str,
        default="/opt/ml/processing/input/",
        help="path to save input data to",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/opt/ml/processing/output/",
        help="path to save output data to",
    )
    args, _ = parser.parse_known_args()
    param_dict = copy.copy(vars(args))

    logger.info(f"Using arguments {json.dumps(param_dict, indent=2)}")

    logger.info("loading files")
    X_image_data = np.load(os.path.join(param_dict["input_path"], "image_file.npy"))
    X_meta_data = np.load(os.path.join(param_dict["input_path"], "meta_data.npy"))
    y = np.load(os.path.join(param_dict["input_path"], "y.npy"))
    logger.info("files loaded")

    # create train and test datasets

    (
        X_image_train,
        X_image_test,
        X_meta_train,
        X_meta_test,
        y_train,
        y_test,
    ) = train_test_split(X_image_data, X_meta_data, y, test_size=0.2)
    logger.info("split data")

    # fit model
    input_shape = (128, 108, 1)
    batch_size = 256
    num_epoch = 100

    logger.info("fitting model")
    pitch_model = model.PitchModel(image_shape=input_shape)
    pitch_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    early_stopping = model.EarlyStoppingWithThreshold(
        monitor="val_accuracy", patience=10, threshold=0.6
    )

    model_log = pitch_model.fit(
        x=[X_image_train, X_meta_train],
        y=y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        validation_data=([X_image_test, X_meta_test], y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    print(pitch_model.summary())

    pitch_model.save(os.path.join(param_dict["output_path"], "model"))
    print("model saved")
