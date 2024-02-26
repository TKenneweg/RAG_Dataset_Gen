
from tqdm import tqdm
import requests
import wikipedia
import pickle
import os
from openai import OpenAI
import json
import time
import pickle
from dotenv import load_dotenv

class WikiScrapper:
    def __init__(self, path):
        load_dotenv()
        self.path = path


    #outputs A_r
    def scrapeAndSaveArticles(self, n=10):
        print(f"Scraping {n} articles")
        n_it = n//500 +1
        n_ret = n if n<500 else 500
        dictlist = []
        for i in tqdm(range(n_it)):
            titles = self.get_random_wikipedia_pages(n_ret)
            for title in titles:
                elem = {}
                try:
                    #this will sometimes fail, so we need to catch the exception
                    page = wikipedia.page(title, auto_suggest=False) 
                except:
                    continue
                elem["title"]= title
                elem["content"] = page.content
                elem["url"] = page.url
                # print(elem["title"])
                # print(elem["content"])
                # print("\n\n ########################### \n\n")
                if len(elem["content"]) > 1000: #only save article if it has a minmum length
                    dictlist.append(elem)

        #save dictlist to disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(f"{self.path}/A_r.pkl", "wb") as file:
            pickle.dump(dictlist, file)
        print(f"Saved {len(dictlist)} articles to file {self.path}/A_r.pkl")
        print(f"Missed {n - len(dictlist)} articles")


    #outputs A_d without questions
    def filterforDate(self,filename, date= "2021-10-01T00:00:00Z"):
        with open(filename,"rb") as file:
            A_r = pickle.load(file)
        print(f"Read {len(A_r)} articles from {filename}")
        # A_r = A_r[:10]

        dictlist = []
        for elem in A_r:
            date= self.get_wiki_page_creation_date(elem["title"])
            # print(date)
            # print(elem["title"])
            if date > "2021-10-01T00:00:00Z":
                elem["date"] = date
                dictlist.append(elem)

        
        #save dictlist to disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(f"{self.path}/A_d.pkl", "wb") as file:
            pickle.dump(dictlist, file)
        print(f"Saved {len(dictlist)} articles to file {self.path}/A_r.pkl")


    #adds questions to A_d
    def genQs(self, filename):
        with open(filename,"rb") as file:
            A_d = pickle.load(file)
        print(f"Read {len(A_d)} articles from {filename}")
        # A_d = A_d[:10]

        dictlist = []
        for elem in tqdm(A_d):
            elem["question"] = self.generate_question(elem["title"] + elem["content"])
            dictlist.append(elem)

        
        #save dictlist to disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # with open(f"{self.path}/A_d.pkl", "wb") as file:
        with open(filename, "wb") as file:
            pickle.dump(dictlist, file)
        print(f"Saved {len(dictlist)} articles to file {filename}")


    def filterForRecentness(self, filename):
        with open(filename,"rb") as file:
            A_d = pickle.load(file)
        print(f"Read {len(A_d)} articles from {filename}")
        # A_d = A_d[:10]
        openaiclient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        function_descriptions = [
            {
                "name": "afterDate",
                "description": "Set if an article was created after a certain date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recentness": {
                            "type": "boolean",
                            "description": "Describes if most of the information in the article is about things that happened after september 2021.",
                        },
                    },
                    "required": ["recentness"],
                },
            }
        ]

        def exponential_backoff(max_retries=5, base_delay=1):
            # print(messages)
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    chatCompletion = openaiclient.chat.completions.create(
                        model="gpt-4-0613",
                        temperature=0, 
                        messages=messages,
                        functions=function_descriptions,
                        function_call={"name": "afterDate"})
                    params = json.loads(chatCompletion.choices[0].message.function_call.arguments)
                    return params["recentness"]
                except Exception as e:
                    print(f"Error occurred: {e}")
                    retries += 1
                    time.sleep(delay)
                    delay *= 2
            return None
        
        dictlist = []
        for elem in tqdm(A_d):
            messages = []
            article = elem["title"] + elem["content"]
            article = article[:15000]
            messages.append({"role": "system", "content": f"""Is most of the information in the following article about things that happened after september 2021? \n\n 
                             {article}"""})
            recent = exponential_backoff()
            if recent:
                dictlist.append(elem)


        # Save A_f to disk
        foldername = filename.split("/")[0]
        with open(f'{foldername}/A_f.pkl', 'wb') as file:
            pickle.dump(dictlist, file)
        print(f"Saved {len(dictlist)} articles to file {foldername}/A_f.pkl")


############################################ utility code ############################################

    def get_random_wikipedia_pages(self,n=1):
        URL = "https://en.wikipedia.org/w/api.php"

        PARAMS = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": "0",  # 0 corresponds to Wikipedia articles
            "rnlimit": f"{n}"  # Number of random pages to return
        }

        response = requests.get(url=URL, params=PARAMS)
        data = response.json()
        random_page_titles = [page['title'] for page in data['query']['random']]
        return random_page_titles
    

    def get_wiki_page_creation_date(self,page_title):
        URL = "https://en.wikipedia.org/w/api.php"

        PARAMS = {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvdir": "newer",  # This orders the revisions from oldest to newest
            "rvlimit": "1",  # We only want the first revision
            "format": "json"
        }

        response = requests.get(URL, params=PARAMS)
        data = response.json()

        # The page ID is needed to access the revisions
        page_id = next(iter(data["query"]["pages"]))
        creation_date = data["query"]["pages"][page_id]["revisions"][0]["timestamp"]

        return creation_date
    

    def generate_question(self,contents):
        contents = contents[:15000]
        openaiclient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        messages = [{"role": "system", "content": f"Generate a creative question about the contents of the following text: {contents}"}]
        
        # Code to feed contents to GPT-4 and generate questions
        chatCompletion = openaiclient.chat.completions.create(model="gpt-4-0613", messages=messages, temperature=1)
        response = chatCompletion.choices[0].message.content
        return response