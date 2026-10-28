"""Autonoms."""

import os

# Autonoms sends nothing to any tracing service. But LangGraph installs the LangSmith client
# library, which starts uploading prompts and code to a third party if one of these variables
# is set in your shell. Force them off, before any LangChain/LangGraph code loads.
for _name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_name] = "false"
