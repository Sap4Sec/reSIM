# reSIM
Code for the "Neural Re-ranking for Binary Function Similarity Search" paper


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

and decompress the .tar.zst archives of BinCorp, MultiComp and BinPool datasets.

## Dataset

Resources for running the experiments as well as the fine-tuned reSIM model are available <a href="https://zenodo.org/records/18505205?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc5Njc1OGUzLWU2ODEtNDg5Zi05ZjVhLWU3ZDY4MTdkZGZjOCIsImRhdGEiOnt9LCJyYW5kb20iOiIzYzdlOWY4ODMwOWUyMTI0ZTllZGU0ZDc1Y2NhNDdhNiJ9.cBZQTVziY_IJvmAVRBIBM5L_BGDDtPcGkoOWvfGr4IG6Aaj5jvwoIf2WJsCPou3BI-5OC9wt1mMJKDkvCPl-uA">here</a>

