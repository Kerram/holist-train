docker build -f Dockerfile -t bert_perf_gpu .

docker run --gpus device=0 -v /home/zpp/deephol-data:/home/data \
-v /home/zpp/models_after_fine_tuning/bert_perf_gpu:/home/model -it bert_perf_gpu /bin/bash
