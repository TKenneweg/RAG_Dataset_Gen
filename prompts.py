system_prompt_nrag = """You task is to answer questions. You will be provided with relevant information to answer the question. 
The information are extracts from wikipedia articles. The title of the article is provided at the top of each information block to give you context."""

system_prompt_barag = """You task is to answer questions. For each question you can choose wheter you want to query a database to be provided with addtional information.
If it is at all possible to answer a questions without querying the database you should do so in order to save tokens.
If you choose to query the database the information provided are extracts from wikipedia articles. 
The title of the article is provided at the top of each information block to give you context."""

system_prompt_barag_new = """You task is to answer questions. Sometimes but not always you will be provided with relevant information to answer the question. 
The information are extracts from wikipedia articles. The title of the article is provided at the top of each information block to give you context."""

system_prompt_baseline = ""


content_prompt = """{chunks}"""

user_prompt = """{query}"""