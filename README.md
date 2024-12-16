# Multi-LLM Agent System

This project implements a modular multi-LLM agent system where different language models handle specific cognitive tasks. The system demonstrates how to create a sophisticated AI agent by combining the strengths of various LLMs through OpenRouter.

## Features

### Core Features
- **Multi-Model Architecture**: Each cognitive task is handled by a specialized LLM:
  - Reasoning: Deep analysis using GPT-4 Preview
  - Planning: Task breakdown using Claude-3.5 Sonnet
  - Execution: Action generation using Claude-3.5 Haiku
- **Web Interface**: Modern Streamlit-based UI for easy interaction
- **Context Awareness**: Support for adding context to improve task understanding
- **History Management**: Track and export interaction history
- **Image Support**: Process and analyze images alongside text

### Technical Features
- **Intelligent Caching**: Caches responses for improved performance and reduced API costs
- **Rate Limiting**: Built-in rate limiting for API request management
- **Async Processing**: Asynchronous operations for better performance
- **Error Handling**: Robust error handling with retries and fallbacks
- **Type Safety**: Full type hints and Pydantic models for reliability

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and add your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   SITE_URL=your_site_url  # Optional
   APP_NAME=your_app_name  # Optional
   ```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

This will open a web interface where you can:
- Add optional context
- Enter tasks for the agent
- View the agent's analysis, plans, and actions
- Track history of all interactions

### Python API

You can also use the system programmatically:
```python
from multi_llm_agent.config import create_default_config
from multi_llm_agent.agent import MultiLLMAgent

async def main():
    # Create the agent with configuration
    config = create_default_config()
    
    # Use async context manager for proper cleanup
    async with MultiLLMAgent(config) as agent:
        # Add context if needed
        agent.add_context({
            "domain": "software development",
            "expertise_level": "intermediate"
        })
        
        # Process a task with timeout
        response = await asyncio.wait_for(
            agent.process("Your task here"),
            timeout=120  # 2 minute timeout
        )
        
        # Access the results
        print("Analysis:", response.thought_process)
        print("\nPlan:")
        for step in response.plan:
            print(f"- {step}")
        print("\nAction:", response.action)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Architecture

The system consists of the following components:

- `agent.py`: Main agent orchestrator that coordinates between modules
- `llms/`: Individual LLM modules
  - `reasoning.py`: Deep analysis using GPT-4 Preview
  - `planner.py`: Task planning using Claude-3.5 Sonnet
  - `executor.py`: Action generation using Claude-3.5 Haiku
  - `base.py`: Base LLM module with shared functionality
  - `cache_control.py`: Caching system for API responses
  - `rate_limiter.py`: Rate limiting for API requests
  - `image_handler.py`: Image processing utilities
- `config.py`: Configuration management with environment variables
- `app.py`: Streamlit web interface with history tracking

_Last updated: December 16, 2024_
