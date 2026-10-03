#!/bin/sh
# Omnigent Robotics Lab launcher (macOS, local only).
#   ./lab.sh setup     install deps, generate the benchmark, freeze the eval, make policies importable by Omnigent
#   ./lab.sh run       pursue goals/<goal>.yaml autonomously (PI + 8 specialists), resuming from the record
#   ./lab.sh run -p "..."   same lab, your own instruction
#   ./lab.sh baseline [-p ...]   single-agent control (same tools and budget, separate record)
#   ./lab.sh report    export the record and build REPORT.md
set -e
cd "$(dirname "$0")"
export LAB_ROOT="$PWD"
case "$1" in
  setup)
    uv sync --python 3.12
    uv tool install --python 3.12 omnigent --with-editable "$LAB_ROOT"   # policies run inside Omnigent
    [ -d scenes/dev ] || uv run python -m lab.scenes
    [ -f record/frozen_eval.sha256 ] || uv run python -m lab.runner freeze
    [ -d "${SUMO_DIR:-$HOME/src/sumo}" ] || echo "optional: clone rai-opensource/sumo to ~/src/sumo and run 'pixi install && pixi run build'"
    ;;
  run)
    shift
    # No arguments: pursue the goal file autonomously. With -p "...": your own instruction instead.
    [ $# -eq 0 ] && set -- -p "Pursue the research goal autonomously: read the goal, resume from the research record, and run discovery cycles back to back until a stop condition holds."
    exec omnigent run agents/assembly_lab "$@"
    ;;
  baseline)
    shift
    exec omnigent run agents/single_agent_baseline "$@"
    ;;
  report)
    uv run python -m lab.report
    ;;
  *)
    sed -n 2,7p "$0"
    ;;
esac
