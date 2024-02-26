This is repo for the paper "Retrieval Augmented Generation Systems: Automatic Dataset Creation, Evaluation and Boolean Agent Setup". If you use this repo please cite the paper.


This repo provides functionality to create datasets from Wikipedia, which are not or only partially contained in the LLM training set. 
Functionality to automatically evaluate different RAG systems using LLM evaluation is also included. 

The file main.py contains all steps to create, answer and evaluate a dataset. 
There are 3 main classes: 

1. WikiScrapper: Responsible to scrape articles of Wikipedia and filter them. 
2. Chatbot: Implement your RAG system here. Naive RAG is already implemented. 
3. Evaluator: Used to evaluate questions/article - answer pairs. Evaluates for truthfulness and relevance. 


Furthermore, embedd.py provides a ready-made script to transform your Wikipedia dataset into a chroma vector db for RAG. 

