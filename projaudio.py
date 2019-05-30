from __future__ import unicode_literals
import glob
import librosa
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
import youtube_dl
            
import csv
import os
import numpy as np

def scrape_songs():
    '''
    Download around 100 songs from each genre playlist (some may fail if the song was deleted). 
    Store them inside separate sub-folders at: "./songs/_genre_/song_name.mp3" 
    '''
    
    genres = {
        'country': 'https://www.youtube.com/watch?v=zXDAYlhdkyg&list=PL3oW2tjiIxvQW6c-4Iry8Bpp3QId40S5S', 
        'jazz': 'https://www.youtube.com/watch?v=RPfFhfSuUZ4&list=PL8F6B0753B2CCA128', 
        'hip_hop': 'https://www.youtube.com/playlist?list=PLAPo1R_GVX4IZGbDvUH60bOwIOnZplZzM', 
        'classical': 'https://www.youtube.com/watch?v=kSE15tLBdso&list=PLRb-5mC4V_Lop8KLXqSqMv4_mqw5M9jjW', 
        'metal': 'https://www.youtube.com/playlist?list=PLfY-m4YMsF-OM1zG80pMguej_Ufm8t0VC', 
        'electronic': 'https://www.youtube.com/watch?v=pvuN_WvF1to&list=PLDDAxmBan0BKeIxuYWjMPBWGXDqNRaW5S'
    }

    for genre, playlist in genres.items():

        path = 'songs/' + genre + '/%(title)s.%(ext)s'

        ydl_opts = {
            'format': 'bestaudio/best',
            'ignoreerrors': True,
            'playlistend': 110,
            'nooverwrites': True,
            'outtmpl': path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([playlist])


def process_songs():
    header = 'name tempo beats chroma_stft rmse spec_cent spec_bw rolloff zcr'.split()
    header += ['mfcc_' + str(i) for i in range(1,12)] + ['label']
    file = open('data.csv', 'w', newline='')

    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        genres = ['country', 'jazz'] #, 'metal', 'hip_hop', 'electronic', 'classical']

    for genre in genres:
        for filename in os.listdir('./songs/' + genre):
            if filename.endswith('.mp3'):
                songname = f'./songs/{genre}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=180, sr=None)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rmse = librosa.feature.rmse(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
                label = genre.title()

                to_append = f'{tempo}, {beats.shape[0]}, {np.mean(chroma_stft)}, {np.mean(rmse)},'
                to_append += f'{np.mean(spec_cent)}, {np.mean(spec_bw)}, {np.mean(rolloff)}, {np.mean(zcr)}'    

                for e in mfcc[1:]:
                    to_append += f',{np.mean(e)}'

                to_append += f',{label}'
                file = open('data.csv', 'a', newline='')

                with file:
                    writer = csv.writer(file)
                    song_row = [filename] + to_append.split(',')
                    writer.writerow(song_row)
            
            
            
            
            
def read_songs(n=None, sr=None, folder=None, suffix=['mp3', 'm4a'], verbose=False):
    '''Loads multiple songs from either the current folder, or a specified folder 
    (which must be contained inside the current folder). 
    
    params:
        n = number of songs to be returned, 
            if not specified, will load all songs in folder
        sr = sample rate, 
            if not specified, will use the default for the original source audio
        folder = 'folder_name' (with out any slashes) 
            if not specified, will search current folder
        suffix = [list, of, file, endings, as, strings]
            defaults to mp3 & m4a files
        verbose = Boolean   
            print summary statement: number of files found, which file types, which folder
    '''
    paths = []
    songs = {}
    
    if folder is None:
        folder = '*.'
    else:
        folder += '/*.'
            
    for end in suffix:
        paths += glob.glob(folder + end)
    
    if n is None:
        n = len(paths)
    
    if verbose:
        print(f"Found {n} file(s) ending with {str(suffix)[1:-1]} in '/{folder[:-2]}' folder.")
    
    for path in paths[:n]:
        if '/' in path:
            songname = path.rsplit('/', 1)[1][:-4]
        else:
            songname = path[:-4]
        audio, sr = librosa.load(path, sr=sr, res_type='kaiser_fast')
        songs[songname] = {'y':audio, 'sr':sr}
        
    return songs


def play_button(y, rate, start_t=0, stop_t=None):
    '''Insert a play button that clips the audio between start and stop times. 
    By default, play the entire audio file.'''
    start = librosa.time_to_samples(start_t, rate)
    
    if stop_t is not None:
        stop = librosa.time_to_samples(stop_t, rate)
    else:
        stop = len(y)
        
    return IPython.display.display(IPython.display.Audio(data=y[start:stop], rate=rate))

    
def chromaplot(y, rate, start_t=0, stop_t=None, play=True, harmonic_input=False):
    
    start = librosa.time_to_samples(start_t)

    if stop_t is not None:
        stop = librosa.time_to_samples(stop_t)
    
    if harmonic_input is False:
        h, p = librosa.effects.hpss(y[start:stop])
    else:
        h = y[start:stop]
        
    C = librosa.feature.chroma_cqt(y=h, sr=rate)
    
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=rate, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if play:
        return play_button(y, rate, start_t, stop_t)
    


    

