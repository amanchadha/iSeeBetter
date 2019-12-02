"""
This file fetches and sanity-checks the dataset so it can be consumed by the algorithm.

- Download the Vimeo90K dataset (get the original test set - not downsampled or downgraded by noise) from:
  http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
- Run this code for LR and HR seperately to form a sorted data folder for convenience
- To delete all the .DS_Store files: find . -name '.DS_Store' -type f -delete

Aman Chadha | aman@amanchadha.com
"""

import argparse, os, sys, shutil, urllib.request, logger
from tqdm import tqdm
import zipfile

################################################### DATASETFETCHER KNOBS ###############################################
# URL to get the the Vimeo90K dataset (get the original test set - not downsampled or downgraded by noise) from
DATASET_URL = "http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip"

# Folder where all the data resides
DATA_FOLDER = "vimeo_septuplet"

# Folder within data where the HR dataset resides
SOURCE_PATH = os.path.join(DATA_FOLDER, "vimeo_test_clean")

# Filename of the dataset
DATASET_FILE = os.path.basename(DATASET_URL)

# Destination folder
DEST_PATH = os.path.join(DATA_FOLDER, "HR")
########################################################################################################################

parser = argparse.ArgumentParser(description='iSeeBetter Dataset Fetcher.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='Print debug spew.')
args = parser.parse_args()

# Initialize logger
logger.initLogger(args.debug)

# Create a data folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    try:
        os.mkdir(DATA_FOLDER)
    except OSError:
        logger.info("Creation of the directory %s failed", DATA_FOLDER)
    else:
        logger.debug("Successfully created the directory: %s", DATA_FOLDER)

class downloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def downloadURL(url, output_path):
    with downloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# If the dataset doesn't exist, download and extract it
if not os.path.exists(SOURCE_PATH):
    # Fetch the dataset if it hasn't been downloaded yet
    if not os.path.exists(SOURCE_PATH + '.zip'):
        downloadURL(DATASET_URL, os.path.join(DATA_FOLDER, DATASET_FILE))

    # Extract it
    logger.info("Extracting: %s", os.path.join(DATA_FOLDER, DATASET_FILE))

    try:
        with zipfile.ZipFile(os.path.join(DATA_FOLDER, DATASET_FILE), 'r') as zipObj:
        # Extract all the contents of zip file in current directory
            zipObj.extractall(DATA_FOLDER)
    except zipfile.BadZipFile:
        # Re-download the file
        downloadURL(DATASET_URL, os.path.join(DATA_FOLDER, DATASET_FILE))
        zipObj.extractall(DATA_FOLDER)

# Recursively remove all the ".DS_Store files"
for currentPath, _, currentFiles in os.walk(SOURCE_PATH):
    if ".DS_Store" in currentFiles:
        os.remove(os.path.join(currentPath, ".DS_Store"))

# Make a list of video sequences
sequencesPath = os.path.join(SOURCE_PATH, "sequences")
videoList = os.listdir(sequencesPath)
videoList.sort()

# Go through each video sequence and copy it over in the structure we need
count = 0
for video in videoList:
   videoPath = os.path.join(sequencesPath, video)
   framesList = os.listdir(videoPath)
   framesList.sort()

   for frames in framesList:
       frames_path = os.path.join(videoPath, frames)
       count += 1
       new_frames_name = count
       des = os.path.join(DEST_PATH, str(new_frames_name))
       logger.info("Creating: %s", des)
       shutil.copytree(frames_path, des)