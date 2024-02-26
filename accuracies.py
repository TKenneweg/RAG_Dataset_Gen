import pickle

def getAverageTruthAndRelevance(dictlist):
    avg_t = 0
    avg_r = 0
    for elem in dictlist:
        avg_t += elem["truthfulness"]
        avg_r += elem["relevance"]
    avg_t /= len(dictlist)
    avg_r /= len(dictlist)
    return avg_t, avg_r

def getTokenCounts(dictlist):
    n_input_tokens = 0
    n_output_tokens = 0
    for elem in dictlist:
        n_input_tokens += elem["n_input_tokens"]
        n_output_tokens += elem["n_output_tokens"]
    return n_input_tokens, n_output_tokens

if __name__ == "__main__":
    filename = "wikirag_090124/A_r_first300_20240125_111901_scored.pkl"
    with open(f"{filename}", "rb") as f:
        dictlist = pickle.load(f)

    print("output for file ", filename)
    #tokens
    n_input_tokens, n_output_tokens = getTokenCounts(dictlist)
    print("n_input_tokens", n_input_tokens)
    print("n_output_tokens", n_output_tokens)
    n_retrieved = 0
    for elem in dictlist:
        if elem["retrieved"]:
            n_retrieved += 1
    print("n_retrieved", n_retrieved)


    #score
    avg_t, avg_r = getAverageTruthAndRelevance(dictlist)
    print("average truthfullness", avg_t)
    print("average relevance", avg_r)
    print("#########################")