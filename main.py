from wikiscrapper import WikiScrapper
from chatbot import Chatbot
import pickle 
from evaluator import Evaluator
import datetime
from tqdm import tqdm


#this saves a pckl and a txt file
def genAnswers(filename,chatbot, description):
    with open(filename, "rb") as f:
        dictlist = pickle.load(f)
    filename = filename[:-4]#remove .pkl from filename

    # dictlist = dictlist[7:17] #good range for testing
    # dictlist = dictlist[14:18] #good range for testing

    dictlist = dictlist[0:300] #for safety

    dictlist_answers = []
    n_not_retrieved = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i,elem in enumerate(tqdm(dictlist)):
        article = elem["title"] + elem["content"]
        article = article[:15000]
        elem["answer"] = chatbot.chat(elem["question"])
        elem["n_input_tokens"] = chatbot.n_input_tokens
        elem["n_output_tokens"] = chatbot.n_output_tokens
        elem["retrieved"] = chatbot.triggerdRetrieval
        elem["context"] = chatbot.context_to_print
        dictlist_answers.append(elem)

        if i%10 == 0: #save in intervals of ten in case something goes wrong
            try: 
                with open(f"{filename}_{current_time}.pkl", "rb") as f:
                    dictlist_answers =  pickle.load(f)+dictlist_answers
            except:
                pass
            with open(f"{filename}_{current_time}.pkl", "wb") as f:
                pickle.dump(dictlist_answers, f)
            dictlist_answers = []
    
    # save data
    try: 
        with open(f"{filename}_{current_time}.pkl", "rb") as f:
            dictlist_answers = pickle.load(f)+dictlist_answers
    except:
        pass
    with open(f"{filename}_{current_time}.pkl" ,"wb") as f:
        pickle.dump(dictlist_answers, f)

    #save description of run
    with open(f"{filename}_{current_time}.txt" ,"w") as f:
        f.write(description)
        
    
    print(f"Saved {len(dictlist_answers)} articles to file {filename}_{current_time}.pkl")

if __name__ == "__main__":
    foldername= "wikirag"
    # scrapper = WikiScrapper(path = foldername)
    # scrapper.scrapeAndSaveArticles(n=200) #this outputs A_r
    # scrapper.filterforDate(foldername+ "/A_r.pkl") #this outputs A_d
    # scrapper.genQs(foldername+ "/A_d.pkl") #add qs to file
    # scrapper.filterForRecentness(foldername+ "/A_d.pkl") #outputs A_f

    chatbot = Chatbot(ragType="Naive")
    description = f"This a naive RAG run."
    genAnswers(f"{foldername}/A_f.pkl",chatbot, description=description)

    # chatbot = Chatbot(ragType="ABARAG")
    # description = f"This a run of advanced boolean agent rag on A_f. Repaired version! less conservative"
    # genAnswers(f"{foldername}/A_f.pkl",chatbot, description=description)


    # chatbot = Chatbot(ragType="Baseline")
    # description = f"This is a baseline run on A_r."
    # genAnswers(f"{foldername}/A_r_first300.pkl",chatbot, description=description)


    # chatbot = Chatbot(ragType="Baseline")
    # description = f"This is a baseline run on A_d."
    # genAnswers(f"{foldername}/A_d.pkl",chatbot, description=description)

    # chatbot = Chatbot(ragType="Baseline")
    # description = f"This is a baseline run on A_f."
    # genAnswers(f"{foldername}/A_f.pkl",chatbot, description=description)

    # chatbot = Chatbot(ragType="Naive")
    # description = f"This is a naive rag run on A_r_first300. to test should not be saved"
    # genAnswers(f"{foldername}/A_r_first300.pkl",chatbot, description=description)


    # evaluator = Evaluator()
    # filename = f"{foldername}/A_r_first300_20240125_111901.pkl"
    # with open(filename, "rb") as f:
    #     dictlist = pickle.load(f)
    # newdictlist = []
    # for elem in tqdm(dictlist):
    #     article = elem["title"] + elem["content"]
    #     article = article[:15000]
    #     elem["truthfulness"],elem["relevance"] = evaluator.evaluate(elem["question"],article,elem["answer"])
    #     # print(elem["truthfulness"], elem["relevance"])
    #     newdictlist.append(elem)
    
    # with open(f"{filename[:-4]}_scored.pkl", "wb") as f:
    #     pickle.dump(newdictlist, f)


