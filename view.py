import pickle

#view pickled files

if __name__ == "__main__":
    with open("wikirag/A_r.pkl", "rb") as f:
        dictlist = pickle.load(f)
    print(len(dictlist))
    for elem in dictlist:
        for key in elem:
            print(key)
            print(elem[key])
        print("\n####################\n")