import asyncio
from config import create_default_config
from agent import MultiLLMAgent

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
    task = """Analyze this multi-LLM agent architecture and suggest improvements 
    for better error handling, caching, and modularity."""
    
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
