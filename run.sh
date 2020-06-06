docker build -f Dockerfile -t bert_perf_final_4520 .

docker run --gpus device=0 -v /home/zpp/deephol-data:/home/data \
-v /home/zpp/models_after_fine_tuning/bert_perf_final_4520:/home/model -it bert_perf_final_4520 /bin/bash

# Inside docker run: python3 experiments.py --dataset_dir=./data/deepmath/deephol/proofs/human/ --model_dir=./model