"""
Command line interface for computer use demo that provides the same functionality as the streamlit app.
"""

import asyncio
import base64
import os
from datetime import datetime
from pathlib import PosixPath
from typing import Any, Optional

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
)

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    SYSTEM_PROMPT,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult
from computer_use_demo.logging_utils import InteractionLogger

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"


class CommandLineInterface:
    def __init__(self):
        self.messages: list[BetaMessageParam] = []
        self.api_key = self._load_api_key()
        self.provider = os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        self.model = PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider]
        self.custom_system_prompt = self._load_system_prompt()
        self.only_n_most_recent_images = 10
        self.hide_images = False
        self.logger = InteractionLogger()

    def _load_api_key(self) -> str:
        """Load API key from file or environment variable."""
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE) as f:
                return f.read().strip()
        return os.getenv("ANTHROPIC_API_KEY", "")

    def _load_system_prompt(self) -> str:
        """Load custom system prompt from file."""
        system_prompt_file = CONFIG_DIR / "system_prompt"
        if os.path.exists(system_prompt_file):
            with open(system_prompt_file) as f:
                return f.read().strip()
        return ""

    def _save_to_storage(self, filename: str, data: str) -> None:
        """Save data to a file in the storage directory."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            file_path = CONFIG_DIR / filename
            with open(file_path, 'w') as f:
                f.write(data)
            file_path.chmod(0o600)
        except Exception as e:
            print(f"Error saving {filename}: {e}")

    def validate_auth(self) -> Optional[str]:
        """Validate authentication credentials."""
        if self.provider == APIProvider.ANTHROPIC:
            if not self.api_key:
                return "Enter your Anthropic API key to continue."
        elif self.provider == APIProvider.BEDROCK:
            import boto3
            if not boto3.Session().get_credentials():
                return "You must have AWS credentials set up to use the Bedrock API."
        elif self.provider == APIProvider.VERTEX:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError

            if not os.environ.get("CLOUD_ML_REGION"):
                return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
            try:
                google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            except DefaultCredentialsError:
                return "Your google cloud credentials are not set up correctly."
        return None

    def output_callback(self, content: BetaContentBlockParam) -> None:
        """Handle content output from the model."""
        if isinstance(content, dict):
            if content["type"] == "text":
                print(f"\nAssistant: {content['text']}")
                self.logger.log_assistant_response(content)
            elif content["type"] == "tool_use":
                print(f"\nTool Use: {content['name']}\nInput: {content['input']}")
                self.logger.log_tool_use(content["name"], content["input"], content["id"])

    def tool_output_callback(self, result: ToolResult, tool_id: str) -> None:
        """Handle tool output results."""
        print("\nTool Output:")
        if result.output:
            print(result.output)
        if result.error:
            print(f"Error: {result.error}")
        if result.base64_image and not self.hide_images:
            print("[Screenshot captured]")
        self.logger.log_tool_result(result, tool_id)

    def api_response_callback(self, request: Any, response: Any, error: Optional[Exception]) -> None:
        """Handle API response and errors."""
        self.logger.log_api_interaction(request, response, error)
        if error:
            if isinstance(error, (APIStatusError, APIResponseValidationError)):
                print(f"\nAPI Error: {error.status_code} - {error.response.text}")
            else:
                print(f"\nAPI Error: {str(error)}")

    async def process_user_input(self, user_input: str) -> None:
        """Process user input and run the sampling loop."""
        message = {
            "role": "user",
            "content": [BetaTextBlockParam(type="text", text=user_input)]
        }
        self.messages.append(message)
        self.logger.log_user_input(message)

        try:
            self.messages = await sampling_loop(
                system_prompt_suffix=self.custom_system_prompt,
                model=self.model,
                provider=self.provider,
                messages=self.messages,
                output_callback=self.output_callback,
                tool_output_callback=self.tool_output_callback,
                api_response_callback=self.api_response_callback,
                api_key=self.api_key,
                only_n_most_recent_images=self.only_n_most_recent_images,
            )
        except Exception as e:
            print(f"\nError: {str(e)}")

    def show_settings(self) -> None:
        """Display current settings."""
        print("\nCurrent Settings:")
        print(f"API Provider: {self.provider}")
        print(f"Model: {self.model}")
        print(f"Only N Most Recent Images: {self.only_n_most_recent_images}")
        print(f"Hide Images: {self.hide_images}")
        if self.custom_system_prompt:
            print(f"Custom System Prompt: {self.custom_system_prompt}")

    def configure_settings(self) -> None:
        """Allow user to configure settings."""
        print("\nConfigure Settings:")
        
        # Configure API provider
        print("\nAvailable API Providers:")
        for i, provider in enumerate(APIProvider):
            print(f"{i + 1}. {provider.value}")
        choice = input("Select API provider (or press Enter to keep current): ")
        if choice.isdigit() and 1 <= int(choice) <= len(APIProvider):
            self.provider = list(APIProvider)[int(choice) - 1]
            self.model = PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider]

        # Configure model
        new_model = input(f"\nEnter model name (current: {self.model}, press Enter to keep): ")
        if new_model:
            self.model = new_model

        # Configure Anthropic API key if using Anthropic
        if self.provider == APIProvider.ANTHROPIC:
            new_key = input("\nEnter Anthropic API key (press Enter to keep current): ")
            if new_key:
                self.api_key = new_key
                self._save_to_storage("api_key", new_key)

        # Configure image settings
        try:
            n_images = input("\nEnter number of most recent images to keep (press Enter to keep current): ")
            if n_images:
                self.only_n_most_recent_images = int(n_images)
        except ValueError:
            print("Invalid input, keeping current value")

        hide_images = input("\nHide images? (y/n, press Enter to keep current): ").lower()
        if hide_images in ('y', 'n'):
            self.hide_images = hide_images == 'y'

        # Configure custom system prompt
        print("\nEnter custom system prompt (press Enter to keep current, or 'clear' to remove):")
        new_prompt = input()
        if new_prompt:
            if new_prompt.lower() == 'clear':
                self.custom_system_prompt = ""
                self._save_to_storage("system_prompt", "")
            else:
                self.custom_system_prompt = new_prompt
                self._save_to_storage("system_prompt", new_prompt)

    async def run(self):
        """Main interaction loop."""
        print("Computer Use Demo - Command Line Interface")
        print("Type ':help' for available commands")

        # Validate authentication
        if error := self.validate_auth():
            print(f"\nError: {error}")
            return

        while True:
            try:
                user_input = input("\nEnter your request (or command): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith(':'):
                    command = user_input[1:].lower()
                    if command == 'quit' or command == 'exit':
                        break
                    elif command == 'help':
                        print("\nAvailable commands:")
                        print(":help    - Show this help message")
                        print(":config  - Configure settings")
                        print(":settings - Show current settings")
                        print(":clear   - Clear conversation history")
                        print(":quit    - Exit the program")
                    elif command == 'config':
                        self.configure_settings()
                    elif command == 'settings':
                        self.show_settings()
                    elif command == 'clear':
                        self.messages = []
                        print("Conversation history cleared")
                    else:
                        print(f"Unknown command: {command}")
                else:
                    await self.process_user_input(user_input)
                    
            except KeyboardInterrupt:
                print("\nUse :quit to exit")
            except Exception as e:
                print(f"\nError: {str(e)}")
                self.logger.log_error(e)


def main():
    """Entry point for the command line interface."""
    cli = CommandLineInterface()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()