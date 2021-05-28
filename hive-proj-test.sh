#!/bin/bash

APP_NAME="proj"

BIN_PREFIX="./"

DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/proj_movielens"
DATA1=("ml_1000000" "ml_5000000" "ml_full")
#DATA1=("ml_full.bin")

APP_OPTIONS=""

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

for file_name in "${DATA1[@]}"
do
     # prepare output json file name with number of gpus for this run
     JSON_FILE="proj__${file_name}__GPU${NUM_GPUS}"

     #echo \
     $BIN_PREFIX$APP_NAME \
     $DATA_PREFIX/$file_name.bin \
     $APP_OPTIONS \
     "$OUTPUT_DIR/$JSON_FILE.json" \
     > "$OUTPUT_DIR/$JSON_FILE.output.txt"
done
