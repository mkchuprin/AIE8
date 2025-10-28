from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller
import random
import requests

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def get_age_by_name(name: str) -> str:
    """Get the estimated age of a person by their name using the Agify API"""
    try:
        response = requests.get(f"https://api.agify.io?name={name}")
        response.raise_for_status()
        data = response.json()
        
        if data.get("age") is None:
            return f"Could not find age data for the name '{name}'"
        
        return f"The estimated age for the name '{data['name']}' is {data['age']} years old (based on {data['count']} samples)"
    except requests.exceptions.RequestException as e:
        return f"Error calling the Agify API: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # print('ehlloooo')