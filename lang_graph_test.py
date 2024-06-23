# {{{ imports 

import os
from dotenv import load_dotenv
load_dotenv()

from getpass import getpass

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, Runnable, RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.chat_models import ChatPerplexity

from datetime import datetime
# }}} 
# {{{ llm initialize 

os.environ["TAVILY_API_KEY"] = "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"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=GEMINI_API_KEY, temperature=0.0)
# llm = ChatPerplexity(model='llama-3-8b-instruct', temperature=0.0, pplx_api_key='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')
# }}} 
# {{{ custom tools 

# @tool
# def company_financial_web_search(input: str) -> str:
#     """Run web search to find the company's financial reports. If the company does not have publicly available financial reports, find the numbers from blogs and articles."""
#     web_search_tool = TavilySearchResults()
#     docs = web_search_tool.invoke({"query": input})

#     return docs

@tool
def web_search(input: str) -> str:
    """Run web search to research about respective information required for company research."""
    web_search_tool = TavilySearchResults()
    docs = web_search_tool({"query": input})

    return docs

# @tool
# def company_business_model_analysis(input: str) -> str:
#     """Perform company

# @tool
# def private_company_research_supervisor(input: str) -> str:
#     """Validate result if the answer is sufficuent or if more research is needed"""
#     pass
tools = [web_search]
# tools = [company_financial_web_search, company_overview_search, private_company_research_supervisor]
# }}} 
# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=GEMINI_API_KEY, temperature=0.0)
# {{{ assistant prompt

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert private equity researcher skilled at creating factual, informative and detailed research on Private Companies. \nFor every input (company), you need to return 2 things:
a. Company Overview - among other web sources, you can also consider linkedin company profile
b. Company Financials - if company does not have publically announced financial details, share financials through blogs/articles/other reports
c. Company Business Model
            You have two tools that you leverage: (1) company_overview_search and (2) company_financial_web_search."""
            "Your image url is: {image_url} "
            "Current time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
# }}} 
# {{{ create runnable, bind tools 

assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)
# }}} 
# {{{ helper functions 

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
# }}} 
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            # Get any user-provided configs 
            image_url = config['configurable'].get("image_url", None)
            # Append to state
            state = {**state, "image_url": image_url}
            # Invoke the tool-calling LLM
            result = self.runnable.invoke(state)
            # If it is a tool call -> response is valid
            # If it has meaninful text -> response is valid
            # Otherwise, we re-prompt it b/c response is not meaninful
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Graph
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

# Define edges: these determine how the control flow moves
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
    # "tools" calls one of our tools. END causes the graph to terminate (and respond to the user)
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)


# try:
#     # Generate the image data
#     img_data = graph.get_graph(xray=True).draw_mermaid_png()

#     # Save the image data to a file to verify it
#     with open("output_image.png", "wb") as f:
#         f.write(img_data)

#     # Check if the file was saved correctly (optional)
#     with open("output_image.png", "rb") as f:
#         display(Image(f.read()))
# except Exception as e:
#     print(f"An error occurred: {e}")

questions = ['What does Zerodha do', 'What are financials of Zerodha', 'what is the weather in Ahmedabad today?']

import uuid
_printed = set()
image_url = None
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "image_url": image_url,
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

# events = graph.stream(
#     {"messages": ("user", 'NJ, USA company Intuitive Cloud')}, config, stream_mode="values"
# )
# for event in events:
#     _print_event(event, _printed)

output = graph.invoke({"messages": ("user", "NJ, USA company Intuitive Cloud")}, config)

