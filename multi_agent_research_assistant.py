# {{{ imports 
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import functools

from langchain_core.messages import AIMessage

from langgraph.prebuilt import ToolNode

from typing import Literal
# }}} 
# {{{ langsmith keys 
LANGCHAIN_TRACING_V2=os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT=os.getenv('LANGCHAIN_PROJECT')
# }}} 
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# llm = ChatOpenAI(model="gpt-4-1106-preview")
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=GEMINI_API_KEY, temperature=0.0)
# {{{  DEF: create_agent

def create_agent(llm, tools, system_message: str):
    """Create an agent"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert researcher, collaborating with other assistant researchers, all skilled at researching private companies and producing informative, descriptive and factual analysis."
                " Use the provided tools to progress towards researching about the particular company."
                " If you are unable to fully answer, that's okay, other assistant with different tools will"
                " help with where you left off."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)
# }}} 
# {{{ tools 

tavily_tool = TavilySearchResults(max_results=5)
# }}} 
# {{{ CLASS: AgentState 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
# }}} 
# {{{  DEF: agent_node

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }
# }}} 
# {{{ create different agents
research_supervisor_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are the the most senior and experienced private equity company researcher. For the company input, check if there is sufficient information available at each step. If not, make sure there is enough information utilizing other agents."
)
research_supervisor_node = functools.partial(agent_node, agent=research_supervisor_agent, name="research_supervisor")

company_overview_research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are a curious and passionate private equity researcher. You should provide accurate information about the company name input such that it covers the 'Company Overview' part of a research.",
)
research_node = functools.partial(agent_node, agent=company_overview_research_agent, name="company_overview_researcher")

financial_research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are a skilled financial analyst skilled at researching and scraping financial information about private companies through publicly available data. For the company input, provide any relevant financial data about the company. If using tavily_tool, suffix search text with 'company financials'",
)
financial_research_node = functools.partial(agent_node, agent=financial_research_agent, name="financial_researcher")

business_model_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are an expert consultant skilled at understanding business models of different companies. For the company input, provide accurate business model based on analysis."
)
business_model_research_node = functools.partial(agent_node, agent=business_model_agent, name="business_model_researcher")

tools = [tavily_tool]
tool_node = ToolNode(tools)
# }}} 
# {{{ DEF: router

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"
# }}} 
workflow = StateGraph(AgentState)

workflow.add_node("research_supervisor", research_supervisor_node)
workflow.add_node("company_overview_researcher", research_node)
workflow.add_node("financial_researcher", financial_research_node)
workflow.add_node("business_model_researcher", business_model_research_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "research_supervisor",
    router,
    {"continue": "company_overview_researcher", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "company_overview_researcher",
    router,
    {"continue": "financial_researcher", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "financial_researcher",
    router,
    {"continue": "company_overview_researcher", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "business_model_researcher",
    router,
    {"continue": "company_overview_researcher", "call_tool": "call_tool", "__end__": END}
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "company_overview_researcher": "company_overview_researcher",
        "financial_researcher": "financial_researcher",
        "business_model_researcher": "business_model_researcher",
        "research_supervisor": "research_supervisor"
    },
)
workflow.set_entry_point("research_supervisor")
graph = workflow.compile()

# {{{ export workflow 

from IPython.display import Image, display

try:
    img = graph.get_graph(xray=True).draw_mermaid_png()
    with open("multi_agent.png", "wb") as f:
        f.write(img)
    with open("multi_agent.png", "rb") as f:
        display(Image(f.read()))
except Exception as e:
    # This requires some extra dependencies and is optional
    print(f'Exception occured: {e}')
# }}} 


output = graph.invoke({"messages": [HumanMessage(content="""Research for company intuitive.cloud. The research should include the following: 
1. Company Overview
2. Company Financials
3. Company Business Model
Once done, finish.""")]})


# events = graph.stream(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Research for Indian company 'Zerodha' and share relevant company financial information. Once done, finish."
#             )
#         ],
#     },
#     # Maximum number of steps to take in the graph
#     {"recursion_limit": 150},
# )
# for s in events:
#     print(s)
#     print("----")
