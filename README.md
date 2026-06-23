# 🧠 Brain Loader

**MLX Multi-Agent Orchestrator for Apple Silicon**

A system where a "Brain" LLM plans complex software projects, delegates tasks to specialized worker models, and iteratively reviews their work until production quality is reached — all running locally on your MacBook Pro M1 Max.

---

## How It Works

```
┌─────────┐     Creates     ┌─────────────┐
│  BRAIN  │ ──────────────► │ Master Plan │
│ (8B)    │                 │ (30-80 tasks)│
└────┬────┘                 └──────┬──────┘
     │                             │
     │ Writes task_001.md          │
     ▼                             ▼
┌─────────┐     Offload    ┌─────────────┐
│  BRAIN  │ ─────────────► │   SLEEP     │
│         │                │             │
└────┬────┘                └─────────────┘
     │
     │ Load Worker
     ▼
┌─────────┐     Executes   ┌─────────────┐
│ WORKER  │ ──────────────►│ result_001.md│
│(15B/7B) │                │             │
└────┬────┘                └──────┬──────┘
     │                            │
     │ Offload Worker             │
     ▼                            │
┌─────────┐     Load Brain       │
│  BRAIN  │ ◄────────────────────┘
│         │     Reviews & Rates
└────┬────┘
     │
     │ If rating < 6: Creates remediation task
     │ If rating >= 6: Moves to next task
     ▼
   Repeat until all tasks done
     │
     ▼
┌─────────┐
│  FINAL  │  Reviews entire project
│ REVIEW  │  → Complete or more tasks
└────┬────┘
     │
     ▼
  Telegram: "🎉 Project Done!"
```

---

## Hardware Requirements

| Component | Minimum | Recommended (Your Setup) |
|-----------|---------|--------------------------|
| RAM | 16GB | **32GB** (25GB allocated) |
| Chip | Apple Silicon M1 | **M1 Max** |
| Storage | 20GB free | **512GB** |
| OS | macOS 13+ | **macOS 14+** |

## RAM Budget (25GB)

| Model | Size (Q4) | Role |
|-------|-----------|------|
| Llama 3.1 8B | ~4.5GB | **Brain** (planner/reviewer) |
| Nemotron-4 15B | ~9GB | Worker (architecture/reasoning) |
| CodeLlama 7B | ~4.5GB | Worker (code generation) |
| DeepSeek Coder 6.7B | ~4GB | Worker (math/logic code) |
| System + MLX overhead | ~3GB | Reserved |

**Critical Rule:** Only ONE model in RAM at a time. The orchestrator handles hot-swapping.

---

## Installation

### 1. Clone & Setup

```bash
git clone <your-repo>
cd brain_loader
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your Telegram token and chat ID
```

### 3. Download Models

```bash
# Brain
huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

# Workers (choose based on your projects)
huggingface-cli download mlx-community/Nemotron-4-15B-Instruct-4bit
huggingface-cli download mlx-community/CodeLlama-7B-Instruct-MLX-4bit
huggingface-cli download mlx-community/deepseek-coder-6.7b-instruct-MLX-4bit
```

Models are cached to `~/.cache/huggingface/` automatically.

### 4. Get Telegram Credentials

1. Message [@BotFather](https://t.me/botfather) → `/newbot` → copy token
2. Message [@userinfobot](https://t.me/userinfobot) → copy your ID
3. Paste both in `config.yaml`

---

## Usage

### Start a New Project

```bash
python main.py "Build a React Native fitness app with AI meal planner"
```

### Resume a Project

```bash
python main.py --resume
```

### Interactive Mode

```bash
python main.py
# Then type your idea when prompted
```

### List Available Models

```bash
python main.py --list-models
```

---

## Project Output Structure

```
tasks/
├── project_state.json          # Machine-readable state
├── task_list.json              # All tasks with metadata
├── master_plan.md              # Brain's original plan
├── master_log.md               # Execution log with timestamps
│
├── task_001_nemotron_15b.md    # Task assignment for worker
├── task_002_codellama_7b.md
├── task_003_codellama_7b.md
│
├── result_001_nemotron_15b.md  # Worker output + Brain rating
├── result_002_codellama_7b.md
├── result_003_codellama_7b.md
│
└── FINAL_SUMMARY.md            # Complete project summary
```

---

## How the Brain Decides

### Task Creation
- Breaks your app idea into 30-80 granular tasks
- Assigns the right model based on task type
- Adds cross-references (e.g., "ensure task 6 includes frontend")

### Review Process
- Rates worker output 1-10
- **6+** = Pass, move to next task
- **<6** = Create remediation task with specific fixes
- Notes adjustments for future tasks

### Final Review
- Checks all tasks are cohesive
- Identifies gaps
- Creates additional tasks if needed (up to 5 iterations)
- Declares `PROJECT_COMPLETE` only when production-ready

---

## Telegram Notifications

You'll receive messages for:
- 🧠 Project start
- 📋 Master plan complete (task count)
- ⚙️ Each task start (with worker model)
- ✅ Task review complete (with rating)
- 🔧 Remediation tasks created
- 🔍 Final review in progress
- 🎉 **Project complete** with file locations

---

## Customization

### Add New Worker Models

Edit `config.yaml`:

```yaml
workers:
  my_custom_model:
    model_path: "mlx-community/Your-Model-4bit"
    max_tokens: 8192
    temperature: 0.3
    description: "What this model does best"
    ram_estimate_gb: 5.0
```

Then reference it in tasks as `my_custom_model`.

### Adjust Task Granularity

```yaml
tasks:
  min_tasks: 50    # More granular
  max_tasks: 100
```

### Change Brain Model

```yaml
brain:
  model_path: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce worker model sizes
- Increase `gc_sleep_seconds` in config
- Use Q3 quantization instead of Q4
- Close other apps

### Model Download Fails
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download <model> --local-dir ./models
```

### Telegram Not Working
- Verify token format: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
- Chat ID should be numeric (e.g., `123456789`)
- Bot must have sent you at least one message first

### Slow Generation
- Normal for first run (model loading to unified memory)
- Subsequent tasks are faster
- Consider reducing `max_tokens` for workers

---

## Architecture

```
brain_loader/
├── main.py                 # CLI entry point
├── config.yaml             # Your configuration
├── requirements.txt
│
├── core/
│   ├── model_manager.py    # MLX load/offload engine
│   ├── task_manager.py     # Task/result file I/O
│   └── orchestrator.py     # Main Brain ↔ Worker loop
│
├── prompts/
│   ├── brain_system.txt    # Brain personality
│   ├── worker_system.txt   # Worker instructions
│   └── brain_review.txt    # Review criteria
│
└── utils/
    └── telegram_notify.py  # Telegram integration
```

---

## License

MIT

---

Built for late-night coding sessions on a MacBook Pro M1 Max. ☕🌙
