# Multi-LLM Agent System

This project implements a modular multi-LLM agent system where different language models handle specific cognitive tasks. The system demonstrates how to create a sophisticated AI agent by combining the strengths of various LLMs through OpenRouter.

## Features

- Modular LLM architecture with specialized models:
  - **Reasoning**: OpenAI GPT-4 Preview (Deep Analysis)
  - **Planning**: Claude-3.5 Sonnet (Task Planning)
  - **Execution**: Claude-3.5 Haiku (Action Generation)
- Streamlit web interface for easy interaction
- Support for context-aware processing
- History tracking of tasks and responses
- Easy to extend with new LLM modules

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
    # Create the agent
    config = create_default_config()
    agent = MultiLLMAgent(config)
    
    # Add context if needed
    agent.add_context({
        "domain": "software development",
        "expertise_level": "intermediate"
    })
    
    # Process a task
    response = await agent.process("Your task here")
    
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

- `agent.py`: Main agent orchestrator
- `llms/`: Individual LLM modules
  - `reasoning.py`: Deep analysis using GPT-4 Preview
  - `planner.py`: Task planning using Claude-3.5 Sonnet
  - `executor.py`: Action generation using Claude-3.5 Haiku
- `config.py`: Configuration management
- `app.py`: Streamlit web interface
