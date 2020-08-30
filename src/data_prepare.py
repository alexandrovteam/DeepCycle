from itertools import islice
import pandas as pd
import numpy as np
import cv2
from const import DATA_ROOT #GFP_ROOT, CY3_ROOT
from argparse import ArgumentParser
from pathlib import Path


SZ = 20
SZ2 = SZ // 2


def get_frame_average(img, yx, sz, fun=np.median):
    """Calculate `fun` across selected patches"""

    sz2 = sz // 2
    mask = np.zeros_like(img, dtype=bool)
    for y, x in yx:
        left, top = max(0, x - sz2), max(0, y - sz2)
        mask[top: y + sz2, left: x + sz2] = True
    return fun(img[mask])


def make_division_adjusted_tracks(curated_tracks_pth, spots_in_tracks_pth, intensities_pth):
    """Align 1000 curated tracks based on division events"""

    curated_tracks = sorted(pd.read_csv(curated_tracks_pth, header=None).astype(int).values.flatten())
    df = pd.read_csv(spots_in_tracks_pth, na_values='None', delimiter='\t').dropna()
    df = df[df['TRACK_ID'].isin(curated_tracks)]

    frame_names = [f.name for f in GFP_ROOT.glob('*')]
    frame_names = sorted(frame_names, key=lambda s: int(s.split('.')[0][1:]))

    div_frames = dict.fromkeys(curated_tracks)
    rows = []
    for frame_name in frame_names:
        print('Frame', frame_name)
        frame_num = int(frame_name.split('.')[0][1:]) - 1

        row = []
        gfp = cv2.imread(str(GFP_ROOT / frame_name), cv2.CV_16U)
        cy3 = cv2.imread(str(CY3_ROOT / frame_name), cv2.CV_16U)
        dt = df.loc[df['FRAME'] == frame_num, ['TRACK_ID', 'POSITION_X', 'POSITION_Y']].astype(int)
        yx = dt[['POSITION_Y', 'POSITION_X']].values
        gfp_frame_average = get_frame_average(gfp, yx, SZ, fun=np.median)
        cy3_frame_average = get_frame_average(cy3, yx, SZ, fun=np.median)
        row.extend([frame_num, gfp_frame_average, cy3_frame_average])

        for track in curated_tracks:
            dxy = dt[dt['TRACK_ID'] == track]
            if (dxy.shape[0] > 1) and (div_frames[track] is None):  # div_frame is where 2 cells
                div_frames[track] = frame_num
            if dxy.shape[0] < 1:
                time = np.nan  # div_frame
                x, y = np.nan, np.nan
                green_median = np.nan
                red_median = np.nan
                green_mean = np.nan
                red_mean = np.nan
            else:
                time = frame_num
                x, y = dxy[['POSITION_X', 'POSITION_Y']].values[0]
                left, top = max(0, x - SZ2), max(0, y - SZ2)
                green_median = np.median(gfp[top: y + SZ2, left: x + SZ2])
                red_median = np.median(cy3[top: y + SZ2, left: x + SZ2])
                green_mean = np.mean(gfp[top: y + SZ2, left: x + SZ2])
                red_mean = np.mean(cy3[top: y + SZ2, left: x + SZ2])
            row.extend([time, x, y, green_median, red_median, green_mean, red_mean])
        rows.append(row)

    div_frames = {k: 0 if v is None else v for k, v in div_frames.items()}
    columns = [('frame_num',), ('gfp_frame_average',), ('cy3_frame_average',)]
    columns_ = [[(track, 'time'), (track, 'x'), (track, 'y')] +
                [(track, color, fun)
                 for fun in ('median', 'mean')
                 for color in ('green', 'red')]
                for track in curated_tracks]
    columns.extend(tt for t in columns_ for tt in t)
    dfo = pd.DataFrame.from_records(rows, columns=pd.MultiIndex.from_tuples(columns))
    for t in curated_tracks:
        dfo[(t, 'time')] -= div_frames[t]
    dfo.to_csv(intensities_pth, index=False)


def clean_df(spots_in_tracks_pth, statistics_clean_pth):
    """Clean and remove unnecessary columns"""

    df = pd.read_csv(spots_in_tracks_pth, na_values="None", delimiter='\t', header=0,
                     usecols=['ID', 'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME']).dropna().astype(int)
    df.to_csv(statistics_clean_pth, index=False)


def add_mean_std(df, curated_tracks_pth, n_frames=None, verbose=False):
    """Add frame average and standard deviation columns for each channel
    Using curated tracks averages"""

    channels = ['GFP', 'Cy3', 'DAPI', 'BF']

    print(f'Adding averages and standard deviations for {", ".join(channels)} channels')

    curated_tracks = sorted(pd.read_csv(curated_tracks_pth, header=None).astype(int).values.flatten())
    df_curated_tracks = df[df['TRACK_ID'].isin(curated_tracks)]

    for channel in channels:
        if verbose:
            print(channel)
        df[channel + '_average'] = 0
        df[channel + '_std'] = 0

        frame_names = (DATA_ROOT / channel).glob("*")
        for frame_name in islice(frame_names, n_frames):
            frame_num = int(frame_name.name.split(".")[0][1:]) - 1
            if verbose:
                print('Frame', frame_name.name, frame_num)

            img = cv2.imread(str(frame_name), cv2.CV_16U)
            yx = df_curated_tracks.loc[df['FRAME'] == frame_num, ['POSITION_Y', 'POSITION_X']].astype(int).values
            img_average = get_frame_average(img, yx, SZ, fun=np.median)
            img_std = get_frame_average(img, yx, SZ, fun=np.std)

            df.loc[df['FRAME'] == frame_num, channel + '_average'] = img_average
            df.loc[df['FRAME'] == frame_num, channel + '_std'] = img_std

        df[channel + '_std'] = df[channel + '_std'].mean()

    return df


def add_intensities(df, sz=20, n_frames=None, verbose=False):
    """Add cell intensities for GFP and Cy3 channels"""

    print('Adding cell intensities for GFP and Cy3 channels')

    sz2 = sz // 2

    df['GFP_' + str(sz)], df['Cy3_' + str(sz)] = 0, 0
    frame_nums = sorted(df['FRAME'].unique())
    for frame_num in frame_nums[:n_frames]:
        frame_name = f'T{frame_num + 1:0>3}'
        if verbose:
            print('Frame', frame_name)

        gfp = cv2.imread(str(GFP_ROOT / frame_name), cv2.CV_16U)
        cy3 = cv2.imread(str(CY3_ROOT / frame_name), cv2.CV_16U)
        green_average = df.loc[df['FRAME'] == frame_num, 'GFP_average'].iloc[0]
        red_average = df.loc[df['FRAME'] == frame_num, 'Cy3_average'].iloc[0]

        intensities = []
        for x, y in df.loc[df['FRAME'] == frame_num, ['POSITION_X', 'POSITION_Y']].values:
            x, y = max(sz2, x), max(sz2, y)
            gfp_intensity = np.median(gfp[y - sz2: y + sz2, x - sz2: x + sz2]) - green_average
            cy3_intensity = np.median(cy3[y - sz2: y + sz2, x - sz2: x + sz2]) - red_average
            intensities.append([gfp_intensity, cy3_intensity])
        df.loc[df['FRAME'] == frame_num, ['GFP_' + str(sz), 'Cy3_' + str(sz)]] = np.array(intensities)

    return df


def add_classes(df, gfp_intensity_col, cy3_intensity_col, class_col_prefix='', n_green=2, n_red=2):
    """Add class column"""

    print('Adding cell classes')

    df['clsCy3'] = pd.qcut(df[cy3_intensity_col], n_red, labels=False)
    df['clsGFP'] = -1

    for cls in range(n_red):
        df.loc[df['clsCy3'] == cls, 'clsGFP'] = \
            pd.qcut(df.loc[df['clsCy3'] == cls, gfp_intensity_col], n_green, labels=False)

    df[class_col_prefix + f'_cls{n_red}x{n_green}'] = (df['clsCy3'] + df['clsGFP'] * n_red).astype(int)
    df.loc[(df['clsCy3'] == -1) | (df['clsGFP'] == -1), class_col_prefix + f'_cls{n_red}x{n_green}'] = np.nan
    del df['clsCy3'], df['clsGFP']
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--microscopy_images', type=str, default=str(DATA_ROOT),
                        help='Input path to the folder containing microscopy images')
    parser.add_argument('--curated_tracks', type=str, default=str(DATA_ROOT / 'curated_tracks.csv'),
                        help='Input path to the list of single division tracks.')
    parser.add_argument('--spots_in_tracks', type=str, default=str(DATA_ROOT / 'Spots in tracks statistics.csv'),
                        help='Input path to the tracking csv file.')
    parser.add_argument('--statistics_clean', type=str, default=str(DATA_ROOT / 'statistics_clean.csv'),
                        help='Output path to the cleaned tracking csv file.')
    parser.add_argument('--statistics_mean_std', type=str, default=str(DATA_ROOT / 'statistics_mean_std.csv'),
                        help='Output path to cleaned cell tracks with added FUCCI averages.')
    parser.add_argument('--intensities', type=str, default=str(DATA_ROOT / 'intensities.csv'),
                        help='Output path to single division tracks aligned on division events.')
    args = parser.parse_args()

    DATA_ROOT = Path(args.microscopy_images)
    GFP_ROOT = DATA_ROOT / 'GFP'
    CY3_ROOT = DATA_ROOT / 'Cy3'

    make_division_adjusted_tracks(args.curated_tracks, args.spots_in_tracks, args.intensities)
    clean_df(args.spots_in_tracks, args.statistics_clean)
    df = pd.read_csv(args.statistics_clean)
    df = add_mean_std(df, args.curated_tracks, n_frames=None, verbose=True)

    df = add_intensities(df, sz=20, n_frames=None, verbose=True)

    df = add_classes(df, gfp_intensity_col='GFP_20', cy3_intensity_col='Cy3_20',
                     class_col_prefix='sq20', n_green=2, n_red=2)

    df.to_csv(args.statistics_mean_std, index=False, float_format='%.3f')
