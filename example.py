import asyncio
from multi_llm_agent.config import create_default_config
from multi_llm_agent.agent import MultiLLMAgent

async def main():
    # Create the agent with default configuration
    config = create_default_config()
    agent = MultiLLMAgent(config)
    
    # Add some context if needed
    agent.add_context({
        "domain": "software development",
        "expertise_level": "intermediate",
        "preferred_language": "python"
    })
    
    # Example task
    task = """Create a simple web scraping script that extracts article titles 
    from a news website and saves them to a CSV file."""
    
    # Process the task
    response = await agent.process(task)
    
    # Print the results
    print("\nThought Process:")
    print(response.thought_process)
    print("\nPlan:")
    for i, step in enumerate(response.plan, 1):
        print(f"{i}. {step}")
    print("\nAction:")
    print(response.action)

if __name__ == "__main__":
    asyncio.run(main())
