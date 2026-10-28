"""Autonoms."""

import os

# Tracing sends prompts and code to a third party (LangSmith). It is opt-in
# (Stage 11) and must stay off otherwise, even if a stray variable is set in
# your shell, so force it off here, before any LangChain/LangGraph code loads.
for _name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_name] = "false"
