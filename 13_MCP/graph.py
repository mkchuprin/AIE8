from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import asyncio

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def main():
    model = init_chat_model("openai:gpt-4.1")
    
    client = MultiServerMCPClient(
        {
            "mcp-server": {
                "command": "python",
                "args": ["/Users/michaelchuprin/Codes/Personal/AIE8-MCP-Session/server.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}
    
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    
    # Interactive menu loop
    print("\n🤖 AI Assistant with Tools Ready!")
    print("=" * 50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Get estimated age by name")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            name = input("Enter a name: ").strip()
            if name:
                try:
                    query = f"What is the estimated age for someone named '{name}'?"
                    response = await graph.ainvoke({"messages": query})
                    last_message = response["messages"][-1]
                    print("\n" + str(last_message.content))
                    print("-" * 50)
                except Exception as e:
                    print(f"Error: {str(e)}\n")
        
        elif choice == "2":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.\n")

if __name__ == "__main__":
    asyncio.run(main())