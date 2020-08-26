import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm, os

# Path to the folder containing the single channel Hoechst microscopy image used for TrackMate. It is important that
# nothing else is present in this folder.
DAPI_Timelapse_folder = r'PATH/TO/DAPI_FILES/'

# Path to the tracking csv file 'Spots in tracks statistics.csv' generated by TrackMate.
csv_p = r"PATH/TO/Spots in tracks statistics.csv"

# Path of the track curation folder which will contain the visualization of the tracks with a potential cell division.
track_curation_folder = r'PATH/TO/Track_curation/'

# # Define the number of frames to use
# n_frames = 200

df = pd.read_csv(csv_p)

# Loop over all tracks and store the track id with potential division in 'split_track.npy'.
if not os.path.exists(track_curation_folder):
    os.mkdir(track_curation_folder)

if not os.path.exists(track_curation_folder + 'split_track.npy'):
    split_track = []
    for track in tqdm.tqdm(np.unique(df['TRACK_ID'])):
        if track != 'None':
            arr = df[df['TRACK_ID'] == track]['FRAME']
            frames, nID = np.unique(arr, return_counts=True)
            division_frame = np.where(nID >= 2)
            if len(division_frame[0]) > 0:
                if nID[division_frame[0][0]] == nID[-1]:
                    split_track = np.append(split_track, track)
    np.save(track_curation_folder + 'split_track.npy', split_track)
else:
    split_track = np.load(track_curation_folder + 'split_track.npy')

# Track curation: for each track creates images of the detected cells in a folder, to be manually curated for cell division

DAPI_filenames = os.listdir(DAPI_Timelapse_folder)

for track_id in split_track:
    if not os.path.exists(track_curation_folder + '{}'.format(track_id)):
        os.mkdir(track_curation_folder + '{}'.format(track_id))

        print(track_id)

        # Get extreme coordinates of the track for image dimensions
        x_min, x_max = np.int(np.min(df[df['TRACK_ID'] == track_id]['POSITION_X'].values)), \
                       np.int(np.max(df[df['TRACK_ID'] == track_id]['POSITION_X'].values))
        y_min, y_max = np.int(np.min(df[df['TRACK_ID'] == track_id]['POSITION_Y'].values)), \
                       np.int(np.max(df[df['TRACK_ID'] == track_id]['POSITION_Y'].values))

        if np.min([x_min, x_max, y_min, y_max]) > 10:
            for frame in tqdm.tqdm(np.unique(df[df['TRACK_ID'] == track_id]['FRAME'].values)):

                x = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track_id][df['FRAME'] == frame]['POSITION_X'].values])
                y = np.array([np.int(xi) for xi in df[df['TRACK_ID'] == track_id][df['FRAME'] == frame]['POSITION_Y'].values])

                im_bf = plt.imread(DAPI_Timelapse_folder + DAPI_filenames[frame])[y_min - 10:y_max + 10,
                          x_min - 10:x_max + 10]
                im_bf_ct = np.clip(im_bf, np.percentile(im_bf, 1), np.percentile(im_bf, 99))

                fig = plt.figure()
                plt.imshow(im_bf_ct, cmap='gray')
                plt.scatter(x-x_min+10, y-y_min+10, 100, color=[1, 0, 0])
                plt.axis('off')
                fig.patch.set_facecolor((0, 0, 0))
                plt.tight_layout()
                plt.savefig(track_curation_folder + '{}/{}.png'.format(track_id, frame))
                plt.close('all')

# During manual curation store the track id (folder name) as field of the dictionary and the frame index
# (image file name, without the file extension) as values. Store these indices in two separate  dictionaries depending
# whether a single or two divisions have been recorded. Single division tracks will serve as training set, whereas
# double division tracks will serve as validation set.

# For illustration the following dictionaries have been filled with fictive data where the track ids '2' and '48' have
# been manually curated for cell division events which occured at the frame '10' and '5', respectively.
single_division_tracks = {
    '2': [10],
    '48': [5]
}

# In case two divisions have been observed, store the results in a separate dictionary. These tracks will serve as
# validation for the model as these cells have been tracked for a whole cell cycle which is required to compute the
# CC time (%). Similarly, fictive data have been added as an example.
double_division_tracks = {
    '15': [5, 150],
    '482': [12, 135]
}
