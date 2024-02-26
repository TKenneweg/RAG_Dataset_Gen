from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import pickle
from tqdm import tqdm
import numpy as np
from openai import OpenAI
import time

if __name__ == "__main__":
    load_dotenv()
    openai_key= os.getenv("OPENAI_API_KEY")
    openaiclient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])



    with open("wikirag_090124/A_r.pkl", "rb") as f:
        dictlist = pickle.load(f)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1028,
        chunk_overlap  = 48,
        length_function = len,
        add_start_index = False,
    )

    allchunks = []
    for elem in tqdm(dictlist):
        chunks = text_splitter.create_documents([elem["content"]]) #return list of langchain.Document
        for chunk in chunks:
            allchunks.append("Title: " + elem["title"] + "\n" + chunk.page_content)
        # print(len(chunks))
        # print(elem["url"])

    # for chunk in allchunks:
    #     print(chunk)
    #     print("\n####################\n")
    print(len(allchunks))



    chroma_client = chromadb.PersistentClient(path="./data/wiki_090124/chroma")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key,  model_name="text-embedding-ada-002" )

    metaallchunks = []
    count =0
    for i,chunk in enumerate(allchunks):
        if i%1000 == 0:
            metaallchunks.append([])
            count +=1
        metaallchunks[-1].append(chunk)
    print(count)

    collection = chroma_client.create_collection(name="wiki_090124_collection", embedding_function=openai_ef)
    ids = [str(i) for i in range(len(allchunks))]
    for i,chunklist in enumerate(metaallchunks):
        print(i)
        collection.add(documents=chunklist, ids=ids[i*1000:(i+1)*1000])
    
    
    # embeddings = []
    # for i,chunk in enumerate(allchunks):
    #     res = openaiclient.embeddings.create(input=chunk, model="text-embedding-ada-002")
    #     embeddings.append(res.data[0].embedding)
    #     # print(type(res.data[0].embedding[0]))
    #     print(i)
    #     print(i%1000)
    #     if i%1000 == 0 and i!=0:
    #         #sleep a bit to not overload the api
    #         print("sleeping")
    #         time.sleep(60)
    # print(len(embeddings))
    # print(len(allchunks))



    # collection = chroma_client.create_collection(name="wiki_090124_collection", embedding_function=openai_ef)
    # collection.add(documents = allchunks, embeddings=embeddings, ids = [str(i) for i in range(len(allchunks))])

