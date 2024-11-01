#!/bin/bash
set -e

# Make sure we have a secure .Xauthority file
touch $HOME/.Xauthority
chmod 600 $HOME/.Xauthority

# Start Xvfb
./xvfb_startup.sh

# Set display for the command line interface
export DISPLAY=:${DISPLAY_NUM}

echo "✨ Computer Use Demo CLI is ready!"
echo "Type :help for available commands"

# Run the command line interface
python -m computer_use_demo.command_line