from dotenv import load_dotenv
load_dotenv()
from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, MessageGraph

from chains import generation_chain, reflection_chain

REFLECT="reflect"
GENERATE="generate"

def generate_node(state: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    return generation_chain.invoke({"messages": state})

def reflect_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res)]
    
builder = MessageGraph()
builder.add_node(GENERATE,generate_node)
builder.add_node(REFLECT,reflect_node)    
builder.set_entry_point(GENERATE)

def should_continue(state : List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE,should_continue)
builder.add_edge(REFLECT,GENERATE)
graph=builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

input = HumanMessage(content=""" We have moved to Tech 3.0 

- AI will automate up to 75% of diagnostic processes by 2028. 
- By 2030, about 90% of trading decisions will be AI-augmented. 
- Self-driving cars will become a reality by 2030. 

The lost jobs in 2025 in tech are NOT coming back, unfortunately.

""")

res=graph.invoke(input)
print(res)