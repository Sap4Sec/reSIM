# reSIM
Code for the "ReSIM: Re-ranking Binary Similarity Embeddings to Improve Function Search Performance" paper


## Build Docker image

To build the docker image using the provided Dockerfile run

```
docker build -t resafe-img .
```

Then start a container using the script:

```
./run_container.sh
```

## Run function search experiment

We provide the pool extracted from BinCorp as well as pre-computed embeddings for the CLAP model. To run the function search experiments, use the script:

```
launch_experiments.py
```

setting the appropriate configuration settings in <code>config.yaml</code>


To reproduce the tables and plots of the paper using pre-computed results, use the notebook

```
reproduce_results.ipynb
```

## Dataset

Resources for running the experiments as well as the fine-tuned reSIM model are available <a href="">here</a>

