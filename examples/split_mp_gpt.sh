#!/bin/bash

TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=training/gpt/tokenizer/gpt2-vocab.json
MERGE_FILE=training/gpt/tokenizer/gpt2-merges.txt
LOAD_PATH=training/gpt/checkpoints/gpt_345m
SAVE_PATH=training/gpt/checkpoints/gpt_345m_splitted_tp2_pp2

WORLD_SIZE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)) \
                                python tools/split_into_mp_partitions.py \
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
                                --load $LOAD_PATH \
                                --save $SAVE_PATH
