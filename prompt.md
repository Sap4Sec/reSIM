Analyze all the files in the function_search/code folder.  
These should be a pipeline that implements the function search experiment where a query binary function is compared against a pool of binary functions to identify the K more similar functions among those in the pool. This is a 2 stage procedure: first a bi-encoder identifies the top-W functions using an embedding-based strategy (The embeddings are pre-computed), then the result is refined by a cross encoder that performs a second function search to identify the top-K (with K <= W).


Create a new folder and inside write a cleaner version of the code. The main entry point of the pipeline is launch_experiments.py. Remove any mention to the vuln_search experiment. I only use reBERT and reDEEP as rerankers.  
Interview me and ask me questions to understand the code better. 