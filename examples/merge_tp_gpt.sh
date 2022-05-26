#!/bin/bash

TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_SIZE=1

VOCAB_FILE=training/gpt/tokenizer/gpt2-vocab.json
MERGE_FILE=training/gpt/tokenizer/gpt2-merges.txt
LOAD_PATH=training/gpt/checkpoints/gpt_345m_tp4
SAVE_PATH=training/gpt/checkpoints/gpt_345m_merged_tp

WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)) \
                                python tools/merge_mp_partitions.py \
                                --model-type GPT \
                                --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
                                --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
                                --tokenizer-type GPT2BPETokenizer \
                                --vocab-file $VOCAB_FILE \
                                --merge-file $MERGE_FILE \
                                --num-layers 24 \
                                --hidden-size 1024 \
                                --num-attention-heads 16 \
                                --seq-length 1024 \
                                --max-position-embeddings 1024 \
                                --load $CHECKPOINT_PATH \
                                --save $SAVE_PATH
