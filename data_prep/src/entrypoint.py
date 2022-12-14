

from urllib.request import urlopen,urlretrieve

from PIL import Image
import json
import argparse
import os
import copy
import pandas as pd
import joblib
import os, os.path
import json
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/',
                        help='path to read input data from')
    parser.add_argument('--table_path', type=str, default='/opt/ml/processing/table/',
                        help='path to read table data from')
    parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/',
                        help='path to save output data to')
    args, _ = parser.parse_known_args()
    param_dict = copy.copy(vars(args))
    
    print(f"Using arguments {json.dumps(param_dict, indent=2)}")

    image_files = [f for f in listdir(param_dict["input_path"]) if f.endswith('.png')]

    pitch_df = pd.read_csv(os.path.join(param_dict["table_path"],'pitches.csv'))
    pitch_df.sort_values('play_id', inplace=True)

    pitch_ids = [f.split('.')[0] for f in image_files]
    pitch_ids.sort()

    pitch_df = pitch_df.loc[pitch_df['play_id'].isin(pitch_ids)]

    images=[]
    print('converting images...')
    for pitch_id in pitch_df['play_id']:
        # load the image
        path = os.path.join(param_dict["input_path"], pitch_id + '.png')
        image = Image.open(path).convert('L')
        images.append(np.asarray(image).reshape(128, -1, 1).astype(np.float16))

    print('finished converting images')
    pitch_df['label'] = pitch_df['pitch_type'].isin(['CU','SL','CH','KC','FC','EP'])*1

    print('creating one-hot-encoded metadata dataset')
    # create one-hot-encoded metadata dataset
    enc = OneHotEncoder(sparse=False)
    X_meta_data = enc.fit_transform(pitch_df[['batter_name']])

    print('saving processed data')
    with open(os.path.join(param_dict["output_path"], 'batter_encoder.pkl'), 'wb') as f:
        joblib.dump(enc, f)

    pitch_df.to_csv(os.path.join(param_dict["output_path"], 'processed_pitch_table.csv'), index=False)
    np.save(os.path.join(param_dict["output_path"], 'image_file.npy'),np.swapaxes(np.array(images),1,3))
    np.save(os.path.join(param_dict["output_path"], 'meta_file.npy'),np.array(X_meta_data))
