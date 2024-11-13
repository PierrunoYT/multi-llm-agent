from typing import Dict, Optional
import openai
import anthropic
from ..config import LLMConfig

class ReasoningModule:
    """
    Reasoning module responsible for deep understanding and analysis.
    Uses sophisticated models through OpenRouter or direct provider APIs.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openai":
            self.client = openai.Client(api_key=config.api_key)
        elif config.provider == "openrouter":
            self.client = openai.Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key
            )
        elif config.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.api_key)
    
    async def analyze(self, input_text: str) -> str:
        """
        Analyze the input and provide detailed reasoning about it.
        Returns a string containing the thought process and analysis.
        """
        prompt = self._create_reasoning_prompt(input_text)
        
        if self.config.provider in ["openai", "openrouter"]:
            extra_args = {}
            if self.config.provider == "openrouter":
                extra_args["extra_headers"] = {
                    "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                    "X-Title": self.config.extra_config.get("app_name", "")
                }
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a reasoning engine focused on deep analysis and understanding."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **extra_args
            )
            return response.choices[0].message.content
            
        elif self.config.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
    
    def _create_reasoning_prompt(self, input_text: str) -> str:
        """Create a detailed prompt for the reasoning task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        
        return f"""Analyze the following input in detail. Consider:
1. Key concepts and their relationships
2. Underlying assumptions and implications
3. Potential challenges or considerations
4. Relevant context and background information

Context:
{context_str}

Input:
{input_text}

Provide a detailed analysis:"""
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
