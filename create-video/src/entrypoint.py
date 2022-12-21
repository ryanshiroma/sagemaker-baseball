
import argparse
import os
import warnings

import pandas as pd
import subprocess
import numpy as np
import sys
import logging
import os, os.path


import logging
import copy
import sagemaker
import json
from os import listdir
from os.path import isfile, join




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_test", type=float, default=0.3)
    parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/',
                        help='path to read input data from')
    parser.add_argument('--table_path', type=str, default='/opt/ml/processing/table/',
                        help='path to read table data from')
    parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/',
                        help='path to save output data to')
    args, _ = parser.parse_known_args()
    param_dict = copy.copy(vars(args))
    
    print(f"Using arguments {json.dumps(param_dict, indent=2)}")


    timestamps = list(np.load(os.path.join(param_dict["table_path"],'timestamps3.npy')))
    confidence =list(np.load(os.path.join(param_dict["table_path"],'confidence3.npy')))
    preds =list(np.load(os.path.join(param_dict["table_path"],'preds3.npy')))

    pitch_df = pd.read_csv(os.path.join(param_dict["table_path"],'processed_pitch_table_large.csv'))

    pitch_df['pred'] = preds
    pitch_df['timestamp'] = timestamps
    pitch_df['confidence'] = confidence

    pitch_df = pitch_df.loc[pitch_df['date']>='2017-05-01']
    pitch_df['date'] = pd.to_datetime(pitch_df['date']).dt.strftime('%b %d, %Y')
    pitch_df = pitch_df.loc[pitch_df['confidence']>0.05]
    pitch_df = pitch_df.loc[pitch_df['pred']>0.5]


    pitch_df['pitch_rank'] = (pitch_df['ab_number']*1000+pitch_df['player_total_pitches'])
    pitch_df['running_count_game']= pitch_df.groupby('game_pk')['pitch_rank'].rank(method='first',ascending=True).astype(int)
    pitch_df = pitch_df.sort_values(['game_pk','running_count_game'])

    pitch_df['running_count'] = range(len(pitch_df))

    pitch_df.reset_index(inplace=True,drop=True)
    



    for i in range(len(pitch_df)):
        pitch = pitch_df.loc[i]
        input_video_path =  os.path.join(param_dict["input_path"],f'{pitch["play_id"]}.mp4')
        output_video_path =  os.path.join(param_dict["output_path"],f'clip_{pitch["play_id"]}.mp4')

        start = np.max([0.01,((pitch['timestamp']/108)*5)-0.75])
        end = np.min([start+1.5,4.99])
        

        with open('subtitle.txt','w') as f: 
            f.write(f'{pitch["date"]}\n')
            f.write(f'Batter: {pitch["batter_name"]}\n')
            f.write(f'Pitcher: {pitch["pitcher_name"]}\n')
            f.write(f'Inning: {pitch["inning"]}\n')
            f.write(f'Outs: {pitch["outs"]}\n')
            f.write(f'Count: {pitch["balls"]}-{pitch["strikes"]}\n')
            f.write(f'AB Result: {pitch["result"]}\n')
            f.write(f'Season Running Count: {pitch["running_count"]+1}\n')
            f.write(f'Game Running Count: {pitch["running_count_game"]}\n')
            f.write(f'Model Confidence: {pitch["pred"]*100:.0f}\\%')
        command = f"""ffmpeg -ss {start} -to {end} -i {input_video_path} -vf "drawtext=textfile=subtitle.txt:fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:font='Arial':x=(w-text_w)/20:y=(h-text_h)/20" {output_video_path}"""
        subprocess.call(command, shell=True)
        print(f'{i}: {pitch["play_id"]}')


    videos = [f'file {os.path.join(param_dict["output_path"],f"clip_{v}.mp4")}' for v in pitch_df['play_id'].values]

    with open(f'{os.path.join(param_dict["output_path"],"mp4_list.txt")}','w') as f:  
        for v in videos: 
            f.write(v+'\n')

    command = f'ffmpeg -f concat -safe 0 -i {os.path.join(param_dict["output_path"],"mp4_list.txt")} {os.path.join(param_dict["output_path"],"merged.mp4")}'
    print(subprocess.call(command, shell=True))





