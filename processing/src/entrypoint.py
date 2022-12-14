
import argparse
import os
import warnings

import pandas as pd
import subprocess
import numpy as np
from PIL import Image
import sys
import logging
from sklearn.model_selection import train_test_split
import os, os.path
import json
import joblib
# from utils import model
# from utils import processing_functions
import librosa
from librosa.display import specshow
import noisereduce as nr
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging
import copy
import sagemaker
from sklearn.exceptions import DataConversionWarning

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join



warnings.filterwarnings(action="ignore", category=DataConversionWarning)


# logger = logging.getLogger()#'PIL').setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))



if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--split_test", type=float, default=0.3)
    parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/',
                        help='path to save input data to')
    parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/',
                        help='path to save output data to')
    args, _ = parser.parse_known_args()
    param_dict = copy.copy(vars(args))
    
    print(f"Using arguments {json.dumps(param_dict, indent=2)}")



    video_files = [f for f in listdir(param_dict["input_path"]) if f.endswith('.mp4')]

    pitch_df = pd.read_csv(os.path.join(param_dict["input_path"],'pitches.csv'))

    print(pitch_df.head())

    # mel spectrogram settings
    duration = 5
    sr = 44100
    fmax = 2000
    nr_threshold = 0.5
    n_mels=128
    n_fft=8192
    hop_length=2048

    pitch_ids = [f.split('.')[0] for f in video_files]
    
    print(pitch_ids[:10])


    base_image_path =  param_dict["output_path"]
    base_audio_path = '/opt/ml/processing/audio'
    base_video_path = param_dict["input_path"]

    # os.makedirs(base_image_path)
    os.makedirs(base_audio_path)

    for pitch_id in pitch_ids:
        print(pitch_id)

        video_path = os.path.join(base_video_path, pitch_id+'.mp4')
        audio_path = os.path.join(base_audio_path, pitch_id+'.wav')
        image_path = os.path.join(base_image_path, pitch_id+'.png')


        #### extract the audio from the video file
        command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 1 -loglevel quiet -stats {audio_path}"
        print(subprocess.call(command, shell=True))


        #### convert the wav to the mel-spectrogram
        y, sr = librosa.load(audio_path,sr=sr,offset=0,duration = duration)[:sr*duration]

        # remove noise from audio
        reduced_noise_y = nr.reduce_noise(y = y, sr=sr, n_std_thresh_stationary=nr_threshold,stationary=True)

        # extend short clips to the duration in seconds
        full_length = sr*duration
        reduced_noise_y = np.hstack((reduced_noise_y,np.zeros(full_length-len(reduced_noise_y))))
        s = librosa.feature.melspectrogram(y=reduced_noise_y, sr=sr, n_mels=n_mels,n_fft=n_fft,hop_length=hop_length,fmax=fmax)
        fig,ax = plt.subplots(1)
        ys,xs=s.shape
        fig.set_size_inches(xs/400, ys/400)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        specshow(librosa.amplitude_to_db(s,ref=np.max),y_axis='mel', fmax=fmax,x_axis='time',ax=ax,cmap='gray_r')
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(image_path, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')
        plt.cla()
        plt.clf()










    # # read in the available images
    # pitch_ids = [f.split('.')[-2] for f in os.listdir(param_dict['input_path']) if f.split('.')[-1] == 'png' ]
    # pitch_ids.sort()
    # print('read in and sorted the pitch pngs')
    # # read in the pitch table and subset from pitch_ids in the images
    # df = pd.read_csv(os.path.join(param_dict['input_path'],'table','pitch_table.csv'))
    # df = df.loc[df['pitch_id'].isin(pitch_ids)].sort_values('pitch_id')
    # pitch_ids = list(df['pitch_id'])
    # df['label'] = df['pitch'].isin(['CU','SL','CH','KC','FC','EP'])*1
    # df['month'] = pd.to_datetime(df['date'],format='%Y-%m-%d').dt.strftime('%Y-%m')
    # # df = df.loc[df['date']>'2017-01-01']
    # y = df[['label']]
    # print('read in and transformed the pitch table')
           
    #  # create one-hot-encoded metadata dataset
    # enc = OneHotEncoder(sparse=False)
    # X_meta_data = enc.fit_transform(df[['batter','month']])
    # with open(os.path.join(param_dict['output_path'],'metadata_processor.joblib'), 'wb') as f:
    #     joblib.dump(enc, f)

    # np.save(os.path.join(param_dict["output_path"],'meta_data.npy'),X_meta_data)
    # np.save(os.path.join(param_dict["output_path"],'y.npy'),y)

    # print('starting pitch file to numpy transformation')                 
    # images=[]
    # for pitch_id in pitch_ids:
    #     # load the image
    #     path = os.path.join(param_dict["input_path"], pitch_id + '.png')
    #     image = Image.open(path).convert('L')
    #     images.append(np.asarray(image).reshape(128, -1, 1).astype(np.float16))
    # image_data = np.array(images)
    # np.save(os.path.join(param_dict["output_path"],'image_file.npy'),image_data)

    # print('finished pitch file to numpy transformation')
