from typing import Dict, List, Optional
from openai import OpenAI
from ..config import LLMConfig

class PlannerModule:
    """
    Planner module responsible for breaking down tasks and creating action plans.
    Uses Claude-3 through OpenRouter for detailed planning.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key,
                default_headers={
                    "HTTP-Referer": config.extra_config.get("site_url", ""),
                    "X-Title": config.extra_config.get("app_name", "")
                }
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def create_plan(self, input_text: str, context: str) -> List[str]:
        """
        Create a structured plan based on the input and reasoning context.
        Returns a list of steps to execute.
        """
        prompt = self._create_planning_prompt(input_text, context)
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are a strategic planner focused on breaking down tasks into actionable steps."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return self._parse_plan(response.choices[0].message.content)
    
    def _create_planning_prompt(self, input_text: str, context: str) -> str:
        """Create a detailed prompt for the planning task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        
        return f"""Create a detailed, step-by-step plan for the following task.
Consider the analysis provided and break down the task into clear, actionable steps.

Context:
{context_str}

Additional Analysis:
{context}

Task:
{input_text}

Create a numbered list of steps to accomplish this task:"""
    
    def _parse_plan(self, response: str) -> List[str]:
        """Parse the response into a list of steps."""
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Remove numbering and any other formatting
        steps = []
        for line in lines:
            # Remove common numbering patterns (1., 1), Step 1:, etc.)
            step = line.lstrip('0123456789.) ').strip()
            if step and not step.lower().startswith(('here', 'step')):
                steps.append(step)
        
        return steps
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
