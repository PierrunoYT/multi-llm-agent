from setuptools import setup, find_packages

setup(
    name="multi_llm_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "python-dotenv",
        "openai",
        "anthropic"
    ]
)
