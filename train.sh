SAVE_DIR=./penalty_experiment
DEVICE=cuda
N_SAMPLES=3500
BATCH_SIZE=128
LR=7e-4
EMBED_DIM=64

mkdir -p $SAVE_DIR

train_model() {
    local save_dir=$1
    local penalty_weight=$2

    mkdir -p "$save_dir"

    echo "Training with penalty weight $penalty_weight... "
    python train.py \
        --save_dir $save_dir \
        --device $DEVICE \
        --n_samples $N_SAMPLES \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --penalty_weight $penalty_weight \
        --embed_dim $EMBED_DIM
}

NO_PENALTY_DIR="$SAVE_DIR/no_penalty"
PENALTY_DIR="$SAVE_DIR/penalty"

train_model "$NO_PENALTY_DIR" 0.0
train_model "$PENALTY_DIR" 0.25
