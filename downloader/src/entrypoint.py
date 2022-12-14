

from pybaseball import statcast,cache
from urllib.request import urlopen,urlretrieve

from bs4 import BeautifulSoup
import requests
import json
import argparse
import os
import copy
import pandas as pd
import os, os.path
import json




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/opt/ml/processing/input/',
                        help='path to save input data to')
    parser.add_argument('--output_path', type=str, default='/opt/ml/processing/output/',
                        help='path to save output data to')
    args, _ = parser.parse_known_args()
    param_dict = copy.copy(vars(args))
    
    print(f"Using arguments {json.dumps(param_dict, indent=2)}")

    cache.enable()
    raw_table = statcast(start_dt='2017-04-03', end_dt='2017-09-21',team='HOU',)

    raw_table = raw_table.loc[raw_table['home_team']=='HOU']

    game_list = []
    games = raw_table['game_pk'].unique()
    for i,game in enumerate(games):
        print(f'getting game ({i}/{len(games)}): {game}')
        url = 'https://baseballsavant.mlb.com/gf?game_pk='+str(game)
        response = urlopen(url)
        full_data = json.loads(response.read())
        
        batters = full_data['home_batters']
        for b in batters:
            print('    pulling batter: ',b)
            game_df = pd.DataFrame(batters[b])
            game_df['date'] = pd.to_datetime(full_data['gameDate'])
            video_ids = []
            for p in game_df['play_id']:
                url = f"https://baseballsavant.mlb.com/sporty-videos?playId={p}"
                v=None
                try:
                    page = requests.get(url)
                    soup = BeautifulSoup(page.text,"lxml" )
                    v=soup.find('video').source['src'].split('/')[-1].split('.')[0]
                    urlretrieve(f'https://sporty-clips.mlb.com/{v}',f'{param_dict["output_path"]}/{p}.mp4')
                except:
                    print('no video file found for play: ',p)
                video_ids += [v]
                
            game_df['video_id'] = video_ids
            game_list += [game_df]
    df = pd.concat(game_list)

    df.to_csv(os.path.join(param_dict["output_path"],'pitches.csv'))
