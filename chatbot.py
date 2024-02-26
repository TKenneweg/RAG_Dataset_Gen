from openai import OpenAI
import prompts
import os
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
import json

#options for ragType: Naive, BARAG, ABARAG

class Chatbot():
    def __init__(self,model = "gpt-4-0613", dbPath = "./data/chroma", ragType = "Naive", useChatHistory = False) -> None:
        self.ragType = ragType
        self.model = model
        self.useChatHistory = useChatHistory
        self.dbPath = dbPath
        self.usermemory = []
        self.aimemory = []

        if self.ragType == "Baseline":
            self.systemprompt = prompts.system_prompt_baseline
        elif self.ragType == "Naive":
            self.systemprompt = prompts.system_prompt_nrag
        elif self.ragType == "BARAG":
            self.systemprompt = prompts.system_prompt_barag
        elif self.ragType == "ABARAG":
            self.systemprompt = prompts.system_prompt_nrag

        self.user_prompt = prompts.user_prompt
        self.content_prompt = prompts.content_prompt

        load_dotenv()
        chroma_client = chromadb.PersistentClient(path="./data/wiki_090124/chroma")
        emb_func = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ['OPENAI_API_KEY'],model_name="text-embedding-ada-002")
        self.collection = chroma_client.get_collection(name="wiki_090124_collection", embedding_function=emb_func)

        self.openaiclient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        self.context_to_print = None
        self.n_input_tokens =0 
        self.n_output_tokens = 0
        self.triggerdRetrieval = False
        self.maxCharactersPerQuestion = 2000
        self.maxQAChainLength = 3
        self.maxModelTokens = 8000



    def NRAGContext(self,query):
        chunks = self.collection.query(query_texts=query, n_results=5)["documents"][0]
        context = "\n \n"
        for chunk in chunks:
            context += chunk + "\n \n"
        return context

    def BARAGContext(self,query):
        # function_descriptions = [
        #     {
        #         "name": "databaseRetrieval",
        #         "description": "Set if a vector database should be queried for additional information to answer the current question.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "retrieve": {
        #                     "type": "boolean",
        #                     "description": """Should be set to false if it is at all possible to answer the questions without additional information.
        #                     Should be set to true if answering the current question is not possible otherwise.""",
        #                 },
        #             },
        #             "required": ["retrieve"],
        #         },
        #     }
        # ]

        function_descriptions_openaistyle = [
            {
                "name": "databaseRetrieval",
                "description": "Get additional information to answer the current question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "retrieve": {
                            "type": "boolean",
                            "description": """Set to true to gain additional information to answer the current question. 
                            If it is possible to answer the current question without additional informaion set to false to save tokens.""",
                        },
                    },
                    "required": ["retrieve"],
                },
            }
        ]

        messages = [{"role": "system", "content": self.systemprompt}]
        for usermemory,aimemory in zip(self.usermemory,self.aimemory):
            messages.append({"role": "user", "content": usermemory})
            messages.append({"role": "assistant", "content": aimemory})
        messages.append({"role": "user", "content": query})

        res = self.openaiclient.chat.completions.create(
                model=self.model,
                temperature=0, 
                messages=messages,
                functions=function_descriptions_openaistyle,
                function_call="auto")
        self.n_input_tokens += res.usage.prompt_tokens
        self.n_output_tokens += res.usage.completion_tokens
    
        print(res.choices[0].finish_reason)
        retrieve = json.loads(res.choices[0].message.function_call.arguments)["retrieve"]
        # print("retrieve ", retrieve)
        if retrieve:
            context = self.NRAGContext(query)
            self.triggerdRetrieval = True
        else:
            context = ""
            self.triggerdRetrieval = False
        return context
    

    def AdvancedBARAGContext(self,query):
        #add to paper the finding that using a query with truthfulness and relevances fails to work. 
        function_descriptions = [
            {
                "name": "moreInformation",
                "description": "Set if you could have answered the last question better with more information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "moreInfo": {
                            "type": "boolean",
                            "description": """Set to true if you could have answered the last question better with more information.""",
                        }
                    },
                    "required": ["moreInfo"],
                },
            }
        ]

        # messages = [{"role": "system", "content": self.systemprompt}]
        messages = [{"role": "user", "content": query}]
        baseAnswers = self.openaiclient.chat.completions.create(model=self.model, 
                messages=messages)
        self.n_input_tokens += baseAnswers.usage.prompt_tokens
        self.n_output_tokens += baseAnswers.usage.completion_tokens
        response = baseAnswers.choices[0].message.content
        messages.append({"role": "assistant", "content": response})

        fc = self.openaiclient.chat.completions.create(
                model=self.model,
                temperature=0, 
                messages=messages,
                functions=function_descriptions,
                function_call={"name": "moreInformation"})
        # print("fc tokens: ", fc.usage.prompt_tokens)
        self.n_input_tokens += fc.usage.prompt_tokens
        self.n_output_tokens += fc.usage.completion_tokens

        retrieve = json.loads(fc.choices[0].message.function_call.arguments)["moreInfo"]
        # print("retrieve ", retrieve)
        # retrieve = True
        if retrieve:
            context = self.NRAGContext(query)
            self.triggerdRetrieval = True
        else:
            context = response
            self.triggerdRetrieval = False
        return context

    def AQRAGContext(self,query):
        pass

    def AQRAGRContext(self,query):
        pass
    

    
    def getMessagesLength(self,messages):
        length = 0
        for message in messages:
            length += len(message["content"])
        return length

    def buildmessages(self,context,query):
        messages = [{"role": "system", "content": self.systemprompt}]
        if (self.useChatHistory):
            for usermemory,aimemory in zip(self.usermemory,self.aimemory):
                messages.append({"role": "user", "content": usermemory})
                messages.append({"role": "assistant", "content": aimemory})

        content_pompt= self.content_prompt.format(chunks=context)
        user_prompt = self.user_prompt.format(query=query)
        messages.append({"role": "user", "content": content_pompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages


    def chat(self,query, gtcontext= None):
        response = ""
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.context_to_print = ""
        #check if query is empty or only spaces
        strippedquery = query.replace(" ","")
        if strippedquery == "":
            return ""
        if len(query) > self.maxCharactersPerQuestion:
            response = "Thats too long, please ask me something shorter."
        else: 
            try:
                if gtcontext is None:
                    if self.ragType == "Baseline":
                        context = ""
                    elif self.ragType == "Naive":
                        context = self.NRAGContext(query)
                    elif self.ragType == "BARAG":
                        context = self.BARAGContext(query)
                    elif self.ragType == "ABARAG":
                        context = self.AdvancedBARAGContext(query)
                        if not self.triggerdRetrieval:
                            return context #super bad variable naming here 
                    else:
                        print("Invalid ragType")
                        return "Invalid ragType"
                else:
                    context = gtcontext
                self.context_to_print = context
                # context = ""
                # return "hi"
            except Exception as e:
                print(f"An exception ocurred when calling chroma {e}")
                response = "Sorry, somethings wrong, try again later (getContext)."
            else:
                messages = self.buildmessages(context,query)
                #check for too long history (should never happen)
                while self.getMessagesLength(messages) > 3*self.maxModelTokens and len(self.aimemory) > 0:
                    self.usermemory.pop(0)
                    self.aimemory.pop(0)
                    messages = self.buildmessages(context,query)
                try:
                    # return "hi"
                    # print(messages)
                    chatCompletion = self.openaiclient.chat.completions.create(model=self.model, 
                    messages=messages)
                    # print("chatCompletion tokens: ", chatCompletion.usage.prompt_tokens)
                    self.n_input_tokens += chatCompletion.usage.prompt_tokens
                    self.n_output_tokens += chatCompletion.usage.completion_tokens
                    response = chatCompletion.choices[0].message.content

                except Exception as e:
                    print(f"An exception ocurred when calling gpt {e}")
                    response = "Sorry,  somethings wrong, try again later (GPT)."

        self.usermemory.append(query) 
        self.aimemory.append(response)
        if len(self.usermemory) > self.maxQAChainLength:
            self.usermemory.pop(0)
            self.aimemory.pop(0)
        return response
        
