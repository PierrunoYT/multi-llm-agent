import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from config import create_default_config
from agent import MultiLLMAgent

# Create history directory if it doesn't exist
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

# Helper functions for history management
def load_history():
    """Load history from the latest history file."""
    try:
        history_files = sorted(HISTORY_DIR.glob("history_*.json"))
        if not history_files:
            return []
        latest_file = history_files[-1]
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        return []

def save_history(history):
    """Save history to a JSON file with timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = HISTORY_DIR / f"history_{timestamp}.json"
        
        # Convert response objects to dictionaries
        serializable_history = []
        for item in history:
            serializable_item = {
                "task": item["task"],
                "context": item["context"],
                "response": {
                    "thought_process": item["response"].thought_process,
                    "plan": item["response"].plan,
                    "action": item["response"].action
                },
                "timestamp": datetime.now().isoformat()
            }
            serializable_history.append(serializable_item)
        
        with open(filename, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        # Keep only last 10 history files
        history_files = sorted(HISTORY_DIR.glob("history_*.json"))
        if len(history_files) > 10:
            for file in history_files[:-10]:
                file.unlink()
    except Exception as e:
        st.error(f"Error saving history: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Multi-LLM Agent System",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Multi-LLM Agent System")
st.markdown("""
This system uses multiple specialized LLM models for different cognitive tasks:
- **Reasoning**: OpenAI GPT-4 Preview (Deep Analysis)
- **Planning**: Claude-3.5 Sonnet (Task Planning)
- **Execution**: Claude-3.5 Haiku (Action Generation)
""")

# Initialize session state
if 'agent' not in st.session_state:
    config = create_default_config()
    st.session_state.agent = MultiLLMAgent(config)

if 'history' not in st.session_state:
    st.session_state.history = load_history()

# Create columns for the interface
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Input")
    # Context input
    st.markdown("### Context (Optional)")
    context_input = st.text_area(
        "Add context for the agent",
        placeholder="Enter any relevant context...",
        key="context",
        height=100
    )
    
    # Task input
    st.markdown("### Task")
    task_input = st.text_area(
        "Enter your task",
        placeholder="Describe what you want the agent to do...",
        key="task",
        height=150
    )
    
    # Process button
    if st.button("Process Task", type="primary"):
        if task_input:
            with st.spinner("Processing..."):
                # Add context if provided
                if context_input:
                    st.session_state.agent.add_context({
                        "user_context": context_input
                    })
                
                # Process the task
                try:
                    # Run the async function in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        st.session_state.agent.process(task_input)
                    )
                    loop.close()
                    
                    # Add to history and save
                    st.session_state.history.append({
                        "task": task_input,
                        "context": context_input,
                        "response": response
                    })
                    save_history(st.session_state.history)
                    
                    # Reset form using st.rerun()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing task: {str(e)}")
        else:
            st.warning("Please enter a task.")

with col2:
    st.subheader("Results")
    
    # Display history in reverse order (most recent first)
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Task {len(st.session_state.history) - i}", expanded=(i == 0)):
            # Task and Context
            st.markdown("#### Task")
            st.write(item["task"])
            
            if item["context"]:
                st.markdown("#### Context")
                st.write(item["context"])
            
            # Response
            st.markdown("#### Analysis")
            st.write(item["response"].thought_process)
            
            st.markdown("#### Plan")
            for j, step in enumerate(item["response"].plan, 1):
                st.write(f"{j}. {step}")
            
            st.markdown("#### Action")
            st.write(item["response"].action)
            
            st.divider()

# Sidebar with additional options
with st.sidebar:
    st.header("Options")
    
    # Clear history button with confirmation
    if st.button("Clear History"):
        if st.button("Confirm Clear History?"):
            st.session_state.history = []
            # Clear history files
            for file in HISTORY_DIR.glob("history_*.json"):
                file.unlink()
            st.rerun()
    
    # Export history button
    if st.button("Export History"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = f"history_export_{timestamp}.json"
            save_history(st.session_state.history)
            with open(HISTORY_DIR / export_file, 'r') as f:
                st.download_button(
                    "Download History",
                    f.read(),
                    file_name=export_file,
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error exporting history: {str(e)}")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This system demonstrates the power of using multiple specialized LLMs for different cognitive tasks. Each model is chosen for its specific strengths:
    
    - **GPT-4 Preview**: Deep understanding and analysis
    - **Claude-3.5 Sonnet**: Structured planning and strategy
    - **Claude-3.5 Haiku**: Efficient action generation
    
    The system coordinates these models to provide comprehensive solutions to complex tasks.
    """)