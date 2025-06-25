from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="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")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    toolsUsed: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """
        You are a research assistant to help generate a research paper.
        Answer the user query and use necessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions}
        """,
        ),
        ("placeholder", "{chat_history}"),
        ('human', "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=False)
query = input("What can I help you research? ")
rawResponse = agentExecutor.invoke({"query": query})
# print(rawResponse.get('output'))

try:
    structuredResp = parser.parse(rawResponse.get("output"))
    print(structuredResp)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", rawResponse)
