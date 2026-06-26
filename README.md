# Brain Loader

Forge is the first iteration of the AI Build Engine. It uses a multi-agent architecture with local LLMs via MLX and optional cloud fallbacks. The Brain model plans tasks, worker models execute them, and the Brain reviews each output before approving. Forge is slow, hot, and precise—it hammers out projects sequentially, task by task, with iterative review.

## Hardware

MacBook Pro M1 Max 32GB (25GB allocated to Forge)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up config
cp config.yaml.example config.yaml
# Edit config.yaml with your tokens

# Run
python main.py "Build a React Native fitness app with AI meal planner"
```

## Architecture

Forge is a multi-agent orchestration framework. It combines local LLMs (via MLX on Apple Silicon) with optional cloud API fallbacks to build complex applications autonomously.

### Models

All models are QLoRA-adapted Mistral variants:

| Model | Parameters | Quantization | Role | Est. VRAM | Max Tokens |
|-----------|-----------|-----------|------|-----------|------------|
| forge-brain-q4 | 7B | Q4_K_M | Planning, Architecture, Review | 5 GB | 8192 |
| forge-coder-q4 | 7B | Q4_K_M | Frontend, React, TypeScript | 4.5 GB | 4096 |
| forge-coderx-q6 | 7B | Q6_K | Complex Logic, Backend APIs | 6 GB | 4096 |
| forge-devops-q4 | 7B | Q4_K_M | CI/CD, Dockerfile, Scripts | 4 GB | 2048 |
| forge-security-q5 | 7B | Q5_K_M | Code Review, Security Audit | 5 GB | 4096 |
| forge-docs-q4 | 7B | Q4_K_M | Documentation, README, API Docs | 3.5 GB | 4096 |
| forge-qa-q4 | 7B | Q4_K_M | Testing, Unit Tests, Integration Tests | 4 GB | 4096 |

### Key Components

- **Brain Orchestrator** (`core/orchestrator.py`) — Central controller, maintains context and state
- **Model Manager** (`core/model_manager.py`) — MLX model loading, generation, and memory management
- **Task Manager** (`core/task_manager.py`) — Task decomposition and execution pipeline
- **Telegram Notifier** (`utils/telegram_notify.py`) — Real-time progress updates

## Cloud Fallbacks

| Provider | Model | Cost/1M Tokens | Purpose |
|----------|-------|--------------|---------|
| DeepSeek | deepseek-chat | ~$0.50 | Full fallback |
| Mistral | mistral-small | ~$2.00 | Fast fallback |
| Anthropic | claude-sonnet-4-20250514 | ~$3.00 | Brain model fallback |
| Together | meta-llama/Llama-4-Maverick-17B | ~$0.80 | Budget fallback |

## Project State

Forge maintains project state in `./memory/`:
```
memory/
├── plan.md — Current plan and task list
├── tasks/ — Individual task outputs
├── review.md — Brain review history
├── working_tree.md — Complete project context
└── brain_context.md — Brain's accumulated context
```

## Logs

All logs are written to `./logs/`. Check `logs/forge_*.log` for detailed execution traces.

## License

MIT