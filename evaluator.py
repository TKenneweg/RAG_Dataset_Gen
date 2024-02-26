from chatbot import Chatbot
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import time


#made specific for wikiqa

#relevant metrics: Relevance (R), Truthfulness(T), Token Usage(U) , (Context_Precision)
#combined metric M = R*T/U (maybe do pareto front)
class Evaluator():
    def __init__(self ) -> None:
        self.systemprompt = """Your task is to evaluate answers given by a chatbot. You are provided
        the user query, the chatbot generated answer and a wikipedia article that contains information about the true answer.
        Given this information generate two scores from 1 to 5, where 5 is the best, for the chatbot generated answer,
        concerning relevance and truthfulness. Give a score of 1 for relevance if the answer is that the chatbot doesn't know."""

        load_dotenv()
        self.openaiclient = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        self.function_descriptions = [
            {
                "name": "setEvaluation",
                "description": "Set the answer evaluation for truthfulness and relevance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "truthfulness": {
                            "type": "number",
                            "description": "The truthfulness of the generated answer on a scale of 1 to 5.",
                        },
                        "relevance": {
                            "type": "number",
                            "description": "The relevance of the generated answer on a scale of 1 to 5.",
                        },
                    },
                    "required": ["truthfulness", "relevance"],
                },
            }
        ]





    def evaluate(self,query, article, cachedresponse = None):
        messages = [{"role": "system", "content": self.systemprompt}]
        response = self.chatbot.chat(query=query) if cachedresponse == None else cachedresponse
        # print("Query: ", query)
        # print("Gtsentence: " ,gtsentence)
        # print("Gen Answ: ", response)
        systemmessage = f"""The wikipedia article which contains information about the true answer is given by:\n\n{article}\n\n ######### \n\n
        
        The query is given by:{query}\n\n ######### \n\n
        
        The chatbot generated answer is given by: {response}\n\n ######### \n\n
        
        Generate two scores from 1 to 5 for truthfulness and relevance of the generated answer."""
        messages = [{"role": "user", "content": systemmessage}]
        def exponential_backoff(max_retries=5, base_delay=1):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    chatCompletion = self.openaiclient.chat.completions.create(
                        model="gpt-4-0613",
                        temperature=0, 
                        messages=messages,
                        functions=self.function_descriptions,
                        function_call={"name": "setEvaluation"})
                    params = json.loads(chatCompletion.choices[0].message.function_call.arguments)
                    return params["truthfulness"], params["relevance"]
                except Exception as e:
                    print(f"Error occurred: {e}")
                    retries += 1
                    time.sleep(delay)
                    delay *= 2
            return None

        result = exponential_backoff()
        if result is None:
            print("Maximum retries exceeded")
        return result
    



    