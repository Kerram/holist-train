FROM python:3.5
WORKDIR /home

# Dependencies.
RUN apt-get update && apt-get install -y python3-pip python-dev libc-ares-dev
RUN pip3 install h5py six numpy wheel mock pyfarmhash grpcio tensorflow==1.14.0
RUN pip3 install keras_applications==1.0.6 keras_preprocessing==1.0.5 --no-deps
ENV \
  PYTHON_BIN_PATH=/usr/bin/python3 \
  PYTHON_LIB_PATH=/usr/local/lib/python3.5/dist-packages

# Get repository.
COPY . .
RUN mkdir model

CMD ["python3", "experiments.py", "--dataset_dir=./deephol-data/deepmath/deephol/proofs/human/", "--model_dir=./model"]
