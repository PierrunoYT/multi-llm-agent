from typing import Dict, List, Optional
from pydantic import BaseModel
from llms.reasoning import ReasoningModule
from llms.planner import PlannerModule
from llms.executor import ExecutorModule
from config import AgentConfig

class AgentResponse(BaseModel):
    thought_process: str
    plan: List[str]
    action: str
    
class MultiLLMAgent:
    def __init__(self, config: AgentConfig):
        """Initialize the Multi-LLM Agent with different cognitive modules."""
        self.config = config
        self.reasoning = ReasoningModule(config.reasoning_config)
        self.planner = PlannerModule(config.planner_config)
        self.executor = ExecutorModule(config.executor_config)
        
    async def process(self, input_text: str) -> AgentResponse:
        """Process input through all cognitive modules to generate a response."""
        # First, analyze and understand the input
        reasoning_result = await self.reasoning.analyze(input_text)
        
        # Based on reasoning, create a plan
        plan = await self.planner.create_plan(
            input_text, 
            context=reasoning_result
        )
        
        # Execute the plan and generate action
        action = await self.executor.execute(
            plan=plan,
            context=reasoning_result
        )
        
        return AgentResponse(
            thought_process=reasoning_result,
            plan=plan,
            action=action
        )
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add additional context to all modules."""
        self.reasoning.add_context(context)
        self.planner.add_context(context)
        self.executor.add_context(context)
