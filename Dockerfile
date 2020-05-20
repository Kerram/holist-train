FROM tensorflow/tensorflow:1.13.2-gpu-py3
WORKDIR /home

RUN mkdir -p model/best_eval
RUN mkdir data
 
COPY . .

CMD ["python3", "experiments.py", "--dataset_dir=./data/deepmath/deephol/proofs/human/", "--model_dir=./model"]
