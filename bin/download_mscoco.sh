#!/bin/bash

# Script to download the MSCOCO images and annotations

DATASET_DIR="data/datasets/mscoco/"
SCRATCH_DIR="data/tmp/"

mkdir -p  $DATASET_DIR
mkdir -p  $SCRATCH_DIR

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  unzip -nq "${FILENAME}"
}

cd ${SCRATCH_DIR}

# Download the images.
BASE_IMAGE_URL="http://images.cocodataset.org/zips"

VAL_IMAGE_FILE="val2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE} 
mv "val2017" "../../${DATASET_DIR}"

TEST_IMAGE_FILE="test2017.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TEST_IMAGE_FILE} 
mv "test2017" "../../${DATASET_DIR}"

cd ..
rm -r ${SCRATCH_DIR}
