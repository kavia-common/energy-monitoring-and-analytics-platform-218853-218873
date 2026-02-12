#!/bin/bash
cd /home/kavia/workspace/code-generation/energy-monitoring-and-analytics-platform-218853-218873/energy_monitor_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

