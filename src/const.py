from pathlib import Path
import pandas as pd


PACKAGE_DIRECTORY = Path(__file__).parent.parent
DATA_ROOT = PACKAGE_DIRECTORY / 'data/Timelapse_2019'
GFP_ROOT = DATA_ROOT / 'GFP'
CY3_ROOT = DATA_ROOT / 'Cy3'

DOUBLE_DIVISION_TRACKS_PTH = DATA_ROOT / 'double_division_tracks.csv'
CURATED_TRACKS_TRACKS_PTH = DATA_ROOT / 'curated_tracks.csv'

double_division_tracks = pd.read_csv(DOUBLE_DIVISION_TRACKS_PTH, index_col='track').to_dict(orient='index')
double_division_tracks = {k: [v['start'], v['stop']] for k, v in double_division_tracks.items()}
curated_tracks = sorted(pd.read_csv(CURATED_TRACKS_TRACKS_PTH, header=None).astype(int).values.flatten())
