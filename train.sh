SAVE_DIR=.
DEVICE=cuda
N_SAMPLES=3500
BATCH_SIZE=128
LR=7e-4

echo "Training with no penalty... "
python train.py \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --penalty_weight 0.0 \
    --model_name no_penalty

echo "Training with penalty... "
python train.py \
    --save_dir $SAVE_DIR \
    --device $DEVICE \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --penalty_weight 0.25 \
    --model_name with_penalty