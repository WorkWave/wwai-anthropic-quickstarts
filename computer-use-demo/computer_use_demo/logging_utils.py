"""
Utility functions for logging interactions in the computer use demo.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from anthropic.types import Message
from anthropic.types.beta import BetaContentBlockParam, BetaMessageParam

from .tools import ToolResult

LOGS_DIR = Path("~/.anthropic/logs").expanduser()


def setup_logging():
    """Ensure the logs directory exists."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_log_filename() -> str:
    """Generate a timestamp-based log filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"interaction_log_{timestamp}.jsonl"


class LogEntry:
    """Represents a single log entry for the interaction."""
    
    def __init__(
        self,
        entry_type: str,
        content: Any,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.entry_type = entry_type
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the log entry to a dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.entry_type,
            "content": self._serialize_content(),
            "metadata": self.metadata
        }

    def _serialize_content(self) -> Any:
        """Serialize the content based on its type."""
        if isinstance(self.content, (BetaMessageParam, dict)):
            return self.content
        elif isinstance(self.content, Message):
            return self.content.model_dump()
        elif isinstance(self.content, ToolResult):
            return {
                "output": self.content.output,
                "error": self.content.error,
                "base64_image": bool(self.content.base64_image),  # Just log if image exists
                "system": self.content.system
            }
        return str(self.content)


class InteractionLogger:
    """Handles logging of all interactions in the computer use demo."""

    def __init__(self):
        setup_logging()
        self.log_file = LOGS_DIR / get_log_filename()
        self.session_metadata = {
            "start_time": datetime.now().isoformat(),
            "platform": os.uname().sysname,
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        }

    def log_entry(
        self,
        entry_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a single entry to the log file."""
        entry = LogEntry(
            entry_type=entry_type,
            content=content,
            metadata=metadata
        )
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')

    def log_user_input(self, message: Union[str, BetaMessageParam]) -> None:
        """Log user input."""
        self.log_entry("user_input", message)

    def log_assistant_response(
        self,
        response: Union[str, BetaContentBlockParam, Message]
    ) -> None:
        """Log assistant's response."""
        self.log_entry("assistant_response", response)

    def log_tool_use(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_id: str
    ) -> None:
        """Log tool usage."""
        self.log_entry(
            "tool_use",
            {"name": tool_name, "input": tool_input},
            {"tool_id": tool_id}
        )

    def log_tool_result(self, result: ToolResult, tool_id: str) -> None:
        """Log tool execution result."""
        self.log_entry(
            "tool_result",
            result,
            {"tool_id": tool_id}
        )

    def log_error(self, error: Exception) -> None:
        """Log an error."""
        self.log_entry(
            "error",
            {
                "type": error.__class__.__name__,
                "message": str(error)
            }
        )

    def log_api_interaction(
        self,
        request: Any,
        response: Any,
        error: Optional[Exception] = None
    ) -> None:
        """Log API request/response interaction."""
        content = {
            "request": request.__dict__ if hasattr(request, '__dict__') else str(request),
            "response": response.__dict__ if hasattr(response, '__dict__') else str(response),
            "error": str(error) if error else None
        }
        self.log_entry("api_interaction", content)