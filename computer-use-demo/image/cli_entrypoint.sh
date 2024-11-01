#!/bin/bash
set -e

# Start Xvfb for basic display functionality
./xvfb_startup.sh

# Set display for the command line interface
export DISPLAY=:1

echo "✨ Computer Use Demo CLI is ready!"
echo "Type :help for available commands"

# Run the command line interface
python -m computer_use_demo.command_line