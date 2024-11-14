import asyncio
from config import create_default_config
from agent import MultiLLMAgent

async def main():
    # Create configuration
    config = create_default_config()
    
    # Initialize agent
    agent = MultiLLMAgent(config)
    
    # Example input
    user_input = "What is the weather like today?"
    
    # Process input
    response = await agent.process(user_input)
    
    # Print results
    print("Thought Process:", response.thought_process)
    print("\nPlan:", "\n".join(response.plan))
    print("\nAction:", response.action)

if __name__ == "__main__":
    asyncio.run(main())
