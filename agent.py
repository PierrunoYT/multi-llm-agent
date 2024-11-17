from typing import Dict, List, Optional, Any, AsyncContextManager
from pydantic import BaseModel, validator
import logging
import asyncio
from contextlib import asynccontextmanager
from llms.reasoning import ReasoningModule
from llms.planner import PlannerModule
from llms.executor import ExecutorModule
from config import AgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    thought_process: str
    plan: List[str]
    action: str
    
    @validator('plan')
    def validate_plan(cls, v):
        if not v:
            raise ValueError("Plan cannot be empty")
        return v

class AgentContext(BaseModel):
    domain: str
    expertise_level: str
    preferred_language: str
    additional_context: Dict[str, Any] = {}

class MultiLLMAgent(AsyncContextManager):
    def __init__(self, config: AgentConfig):
        """Initialize the Multi-LLM Agent with different cognitive modules."""
        self.config = config
        self.reasoning = ReasoningModule(config.reasoning_config)
        self.planner = PlannerModule(config.planner_config)
        self.executor = ExecutorModule(config.executor_config)
        self._context: Optional[AgentContext] = None
        logger.info("MultiLLMAgent initialized with config: %s", config)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources on exit."""
        errors = []
        
        for module in (self.reasoning, self.planner, self.executor):
            try:
                await module.cleanup()
            except Exception as e:
                errors.append(str(e))
                logger.error("Error cleaning up %s: %s", module.__class__.__name__, str(e))
        
        if errors:
            raise Exception(f"Cleanup errors occurred: {'; '.join(errors)}")
        
    async def process(self, input_text: str) -> AgentResponse:
        """Process input through all cognitive modules to generate a response."""
        try:
            logger.info("Processing input: %s", input_text)
            
            # First, analyze and understand the input
            reasoning_result = await self.reasoning.analyze(
                input_text,
                max_retries=2,
                retry_delay=1.0
            )
            logger.debug("Reasoning result: %s", reasoning_result)
            
            # Based on reasoning, create a plan
            plan = await self.planner.create_plan(
                input_text, 
                context=reasoning_result,
                max_retries=2,
                retry_delay=1.0
            )
            logger.debug("Generated plan: %s", plan)
            
            # Execute the plan and generate action
            action = await self.executor.execute(
                plan=plan,
                context=reasoning_result,
                max_retries=2,
                retry_delay=1.0
            )
            logger.debug("Execution result: %s", action)
            
            response = AgentResponse(
                thought_process=reasoning_result,
                plan=plan,
                action=action
            )
            logger.info("Successfully processed input")
            return response
            
        except asyncio.TimeoutError as e:
            logger.error("Operation timed out: %s", str(e))
            raise
        except Exception as e:
            logger.error("Error processing input: %s", str(e))
            raise
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add additional context to all modules."""
        try:
            # Validate context through pydantic model
            validated_context = AgentContext(**context)
            self._context = validated_context
            
            sanitized_context = validated_context.dict()
            logger.info("Adding context to modules: %s", sanitized_context)
            
            # Add context to all modules atomically
            for module in (self.reasoning, self.planner, self.executor):
                try:
                    module.add_context(sanitized_context)
                except Exception as e:
                    logger.error("Failed to add context to %s: %s", 
                               module.__class__.__name__, str(e))
                    # Rollback previous modules
                    self._rollback_context(sanitized_context)
                    raise
                    
        except Exception as e:
            logger.error("Error adding context: %s", str(e))
            raise
            
    def _rollback_context(self, context: Dict[str, str]) -> None:
        """Rollback context changes on error."""
        for module in (self.reasoning, self.planner, self.executor):
            try:
                module.add_context({})  # Reset context
            except Exception as e:
                logger.error("Error rolling back context for %s: %s",
                           module.__class__.__name__, str(e))
