import asyncio
import logging
from typing import Optional
from config import create_default_config
from agent import MultiLLMAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    agent: Optional[MultiLLMAgent] = None
    try:
        # Create the agent with default configuration
        logger.info("Creating agent with default configuration")
        config = create_default_config()
        
        async with MultiLLMAgent(config) as agent:
            # Add some context if needed
            context = {
                "domain": "software development",
                "expertise_level": "intermediate",
                "preferred_language": "python"
            }
            agent.add_context(context)
            
            # Example task
            task = """Analyze this multi-LLM agent architecture and suggest improvements 
            for better error handling, caching, and modularity."""
            
            logger.info("Processing task: %s", task)
            # Process the task with timeout
            try:
                response = await asyncio.wait_for(
                    agent.process(task),
                    timeout=120  # 2 minute timeout
                )
                
                # Print the results
                print("\nThought Process:")
                print(response.thought_process)
                print("\nPlan:")
                for i, step in enumerate(response.plan, 1):
                    print(f"{i}. {step}")
                print("\nAction:")
                print(response.action)
                
            except asyncio.TimeoutError:
                logger.error("Task processing timed out after 120 seconds")
                raise
                
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error("Fatal error: %s", str(e))
        raise
