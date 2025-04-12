from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.website import WebsiteTools 
from pprint import pprint
import logging
import asyncio
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable nested asyncio
nest_asyncio.apply()

# Configure Ollama model
MODEL_NAME = "nemotron-mini:4b"
ollama_model = Ollama(MODEL_NAME)

def create_agents():
    """Create and configure the specialized agents"""
    try:
        # Web scraping agent
        web_agent = Agent(
            name="Web Scraper",
            role="Scrape data from websites",
            model=ollama_model,
            tools=[WebsiteTools(), GoogleSearchTools()],
            description="You are a helpful web scraper who extracts data from websites.",
            instructions=[
                "Given a link by the user, extract the relevant information from the website.",
                "Use the web scraping tool to get the content of web pages.",
                "Use search when you need to find relevant websites first."
            ],
            show_tool_calls=True,
            markdown=True,
            debug_mode=True
        )

        # Finance agent
        finance_agent = Agent(
            name="Finance Agent",
            role="Get financial data",
            model=ollama_model,
            tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
            description="You are a finance expert who provides financial data and analysis.",
            instructions=[
                "Use tables to display financial data.",
                "Provide clear explanations of financial metrics.",
                "Always cite the source of your financial data."
            ],
            show_tool_calls=True,
            markdown=True,
            debug_mode=True
        )

        return web_agent, finance_agent
    except Exception as e:
        logger.error(f"Error creating agents: {e}")
        raise

async def run_analysis(query: str):
    """Run the team analysis with proper error handling"""
    try:
        web_agent, finance_agent = create_agents()
        
        # Team agent
        agent_team = Agent(
            team=[web_agent, finance_agent],
            model=ollama_model,
            description="You are a research assistant that combines web scraping and financial analysis capabilities.",
            instructions=[
                "Use the Finance Agent to get financial data from yfinance.",
                "Use the Web Scraper to find and extract relevant information from websites.",
                "Always include sources for your information.",
                "Use tables to display structured data.",
                "Provide a comprehensive analysis that combines financial data with market insights."
            ],
            show_tool_calls=True,
            markdown=True,
            debug_mode=True
        )

        # Run analysis with streaming
        logger.info("Starting analysis...")
        response = await agent_team.arun(query, stream=True)
        
        async for chunk in response:
            if hasattr(chunk, 'content'):
                print(chunk.content, end='', flush=True)
            else:
                print(chunk, end='', flush=True)
                
        return response

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise

if __name__ == "__main__":
    query = "What's the market outlook and financial performance of AI semiconductor companies?"
    
    try:
        # Ensure Ollama is running
        logger.info(f"Checking Ollama model: {MODEL_NAME}")
        
        # Run the analysis
        result = asyncio.run(run_analysis(query))
        
        # Print formatted results
        print("\n\nFinal Response:")
        print("-" * 80)
        pprint(result)
        
    except Exception as e:
        logger.error(f"Execution error: {e}")
