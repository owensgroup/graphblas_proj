#PARTITION_NAME="dgx2"
#NODE_NAME="rl-dgx2-c24-u16"

PARTITION_NAME="dgxa100_1tb"
NODE_NAME="rl-dgxa-d22-u30"

#NUM_GPUS=16
NUM_GPUS=8

APP_NAME="proj"
BIN_PREFIX="./"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/proj_movielens"
DATA1=("ml_1000000.bin" "ml_5000000.bin" "ml_full.bin")
#DATA1=("ml_full.bin")
APP_OPTIONS=""

OUTPUT_DIR="eval_mgpu/$PARTITION_NAME/$NODE_NAME"
mkdir -p $OUTPUT_DIR

for (( i=1; i<=$NUM_GPUS; i++))
do
    for file_name in "${DATA1[@]}"
    do
        SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"
         $SLURM_CMD $BIN_PREFIX$APP_NAME \
                $DATA_PREFIX/$file_name \
                $APP_OPTIONS \
                > ./$OUTPUT_DIR/${APP_NAME}-${file_name}_GPU$i.txt &
    done
done
