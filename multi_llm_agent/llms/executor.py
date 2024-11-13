from typing import Dict, List, Optional
from openai import OpenAI
from ..config import LLMConfig

class ExecutorModule:
    """
    Executor module responsible for generating specific actions and responses.
    Uses Claude-3.5-haiku through OpenRouter for efficient execution.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def execute(self, plan: List[str], context: str) -> str:
        """
        Execute the plan and generate specific actions or responses.
        Returns a string containing the execution details or response.
        """
        prompt = self._create_execution_prompt(plan, context)
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are an execution engine focused on taking concrete actions based on plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_headers={
                "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                "X-Title": self.config.extra_config.get("app_name", "")
            }
        )
        return response.choices[0].message.content
    
    def _create_execution_prompt(self, plan: List[str], context: str) -> str:
        """Create a detailed prompt for the execution task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        return f"""Execute the following plan based on the provided context and analysis.
Focus on generating specific, actionable outputs for each step.

Context:
{context_str}

Analysis:
{context}

Plan:
{plan_str}

Generate detailed execution steps or responses for this plan:"""
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
