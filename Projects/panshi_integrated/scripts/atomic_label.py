import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add the parent directory of the script to the Python path
# This is needed to import the package if you're running the script directly

from lib.atomic_labeling import (
    tracks_generator
)

if __name__ == "__main__":

    # Step 1:
    #   Input:      ego.csv, obj.csv
    #   Output:     tracks.csv tracks_meta.csv recording.csv

    tracks_generator()

    # tracks_to_labels()