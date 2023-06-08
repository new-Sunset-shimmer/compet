#!/bin/bash
for seed_val in 'roberta-large-openai-detector' 'bertugmirasyedi/deberta-v3-base-book-classification' 'HuggingFaceH4/tiny-random-LlamaForSequenceClassification'
do
echo $seed_val
sed -i "s@model_name = .*@model_name = '$seed_val'@g" /home/yangcw/code/util/config/config.py
python /home/yangcw/code/run.py
wait
done
