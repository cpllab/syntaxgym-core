#!/bin/bash
MODEL_NAME=$1
CORPUS=$2
SEED=$3

SURPRISAL_ROOT="/Users/jennhu/Desktop/lm-zoo/syntaxgym/analysis/surprisal/"
SUITE_ROOT="/Users/jennhu/Desktop/lm-zoo/syntaxgym/test-suites"

mkdir -p "${SURPRISAL_ROOT}/${MODEL_NAME}/${CORPUS}/result"
MODEL="${MODEL_NAME}_${CORPUS}_${SEED}"

suites=($(ls ${SUITE_ROOT}/json))

for suite in "${suites[@]}"
do
    SUITE_NAME=$(echo "$suite" | cut -f 1 -d '.')
    echo $SUITE_NAME
    python agg_surprisals.py \
        -surprisal ${SURPRISAL_ROOT}/${MODEL_NAME}/${CORPUS}/${SUITE_NAME}_${MODEL}.csv \
        -sentences ${SUITE_ROOT}/txt/${SUITE_NAME}.txt \
        -m ${MODEL} \
        -image cpllab/language-models:tiny-lstm \
        -i ${SUITE_ROOT}/json/${SUITE_NAME}.json \
        -o ${SURPRISAL_ROOT}/${MODEL_NAME}/${CORPUS}/result/${SUITE_NAME}_${MODEL}.json
done