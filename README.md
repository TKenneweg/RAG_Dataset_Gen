# Retrieval Augmented Generation Systems: Automatic Dataset Creation, Evaluation and Boolean Agent Setup

This repository is for the paper "Retrieval Augmented Generation Systems: Automatic Dataset Creation, Evaluation and Boolean Agent Setup". If you use this repository, please cite the paper.

## Prerequisites

For everything to work, you need to add an `OPENAI_API_KEY` in a `.env` file to the project!

## Key Features

This repository provides functionality to create datasets from Wikipedia, which are not or only partially contained in the LLM training set. It also includes functionality to automatically evaluate different RAG systems using LLM evaluation.

The file `main.py` contains all steps to create, answer, and evaluate a dataset. There are 3 main classes:

1. **WikiScrapper**: Responsible for scraping articles from Wikipedia, filtering them, and generating questions.
2. **Chatbot**: Implement your RAG system here. Naive RAG is already implemented.
3. **Evaluator**: Used to evaluate questions/article - answer pairs. Evaluates for truthfulness and relevance.

Furthermore, `embedd.py` provides a ready-made script to transform your Wikipedia dataset into a chroma vector db for RAG.

Generally, reading everything in `main.py` starting from `if __name__ == "__main__":` should give you a good understanding of the process.

### Dataset Creation
```
    foldername= "wikirag"
    scrapper = WikiScrapper(path = foldername)
    scrapper.scrapeAndSaveArticles(n=200) #this outputs A_r
    scrapper.filterforDate(foldername+ "/A_r.pkl") #this outputs A_d
    scrapper.genQs(foldername+ "/A_d.pkl") #add qs to file, uses gpt
    scrapper.filterForRecentness(foldername+ "/A_d.pkl") #outputs A_f, uses gpt

```

You can view generated files using view.py. Every .pkl file in this project is a list of dicts.
Different function add different fields to the dicts like "question", "answer", "title", "content", "url"


RAG Process:
```
    chatbot = Chatbot(ragType="Naive")
    description = f"This a naive RAG run."
    genAnswers(f"{foldername}/A_f.pkl",chatbot, description=description)
``` 
Note that genAnswers saves a pkl file that contains the current timestamp to prevent confusion among different RAG runs.
Furthermore a .txt with the same name is generated that contains a description of the run. 
WARNING you need to have a executed embedd.py before this will work!

Evaluation:
```
    evaluator = Evaluator()
    filename = f"{foldername}/A_r_first300_20240125_111901.pkl"
    with open(filename, "rb") as f:
        dictlist = pickle.load(f)
    newdictlist = []
    for elem in tqdm(dictlist):
        article = elem["title"] + elem["content"]
        article = article[:15000]
        elem["truthfulness"],elem["relevance"] = evaluator.evaluate(elem["question"],article,elem["answer"])
        # print(elem["truthfulness"], elem["relevance"])
        newdictlist.append(elem)
    
    with open(f"{filename[:-4]}_scored.pkl", "wb") as f:
        pickle.dump(newdictlist, f)
```
This will generate a pkl file with the same name and an added _scored that contains fields for truthfulness and relevance


**Embedding** 
embedd.py takes a filename in line 19. You should probably pass A_r here. It creates a chroma vector database in a data folder. You can further adjust the subfolder and collection names

**Histograms**



