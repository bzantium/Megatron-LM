#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=training/gpt/data/gpt2_text_document
SAVE_PATH=training/gpt/checkpoints/gpt_345m_tp4
LOAD_PATH=training/gpt/checkpoints/gpt_345m_tp4
VOCAB_FILE=training/gpt/tokenizer/gpt2-vocab.json
MERGE_FILE=training/gpt/tokenizer/gpt2-merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 200 \
       --lr-decay-iters 320000 \
       --save $SAVE_PATH \
       --load $LOAD_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --recompute-method uniform \
       --log-interval 100 \
       --save-interval 100 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16
