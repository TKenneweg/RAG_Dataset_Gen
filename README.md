This is repo for the paper "Retrieval Augmented Generation Systems: Automatic Dataset Creation, Evaluation and Boolean Agent Setup". If you use this repo please cite the paper.

For everything to work you need to add an OPENAI_API_KEY in a .env file to the project!

This repo provides functionality to create datasets from Wikipedia, which are not or only partially contained in the LLM training set. 
Functionality to automatically evaluate different RAG systems using LLM evaluation is also included. 

The file main.py contains all steps to create, answer and evaluate a dataset. 
There are 3 main classes: 

1. WikiScrapper: Responsible to scrape articles of Wikipedia, filter them and generate questions.  
2. Chatbot: Implement your RAG system here. Naive RAG is already implemented. 
3. Evaluator: Used to evaluate questions/article - answer pairs. Evaluates for truthfulness and relevance. 


Furthermore, embedd.py provides a ready-made script to transform your Wikipedia dataset into a chroma vector db for RAG. 

Generally reading everything in main.py starting from if __name__ == "__main__": should give you a good understanding of the process.

Dataset Creation:
    foldername= "wikirag"
    scrapper = WikiScrapper(path = foldername)
    scrapper.scrapeAndSaveArticles(n=200) #this outputs A_r
    scrapper.filterforDate(foldername+ "/A_r.pkl") #this outputs A_d
    scrapper.genQs(foldername+ "/A_d.pkl") #add qs to file, uses gpt
    scrapper.filterForRecentness(foldername+ "/A_d.pkl") #outputs A_f, uses gpt


RAG Process:
    chatbot = Chatbot(ragType="ABARAG")
    description = f"This a run of advanced boolean agent rag on A_r_first300. Repaired version! less conservative"
    genAnswers(f"{foldername}/A_r_first300.pkl",chatbot, description=description)
    
    Note that genAnswers saves a pkl file that contains the current timestamp to prevent confusion among different RAG runs.
    Furthermore a .txt with the same name is generated that contains a description of the run. 

Evaluation:
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


