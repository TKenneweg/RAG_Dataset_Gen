import pickle



if __name__ == "__main__":
    with open("wikirag/A_r.pkl", "rb") as f:
        dictlist = pickle.load(f)
    print(len(dictlist))
    for elem in dictlist:
        print(elem)
        print("\n####################\n")