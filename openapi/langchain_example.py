import json
import os
from langchain import hub
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, AgentExecutor, Tool


# from https://medium.com/mlearning-ai/supercharging-large-language-models-with-langchain-1cac3c103b52

secrets_file = open("secrets.json")
secrets = json.load(secrets_file)
print(secrets)

os.environ["SERPER_API_KEY"] = secrets["SERPER_API_KEY"]
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
os.environ["WOLFRAM_ALPHA_APPID"] = secrets["WOLFRAM_ALPHA_APPID"]

search = GoogleSerperAPIWrapper()
wolfram = WolframAlphaAPIWrapper()

# Choose the LLM to use
llm = OpenAI()

db = SQLDatabase.from_uri("sqlite:///foo_db/Chinook.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Wolfram",
        func=wolfram.run,
        description="Useful for when you need to answer questions about math, science, geography."
    ),
    Tool(
        name="Chinook DB",
        func=db_chain.run,
        description="Useful for when you need to answer questions about Chinook. Input should be in the form of a question containing full context"
    )
]

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
print(prompt)



# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "Who is the prime minister of the UK? Where was he or she born? "
#                                 "How far is their birth place from London?"})

agent_executor.invoke({"input": "What is the full name of the artist who recently released an album called "
                                "'The Storm Before the Calm' and are they in the Chinook database?  "
                                "If so, what albums of theirs are in the Chinook database?"})
