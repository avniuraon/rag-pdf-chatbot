# Agent Instructions for Agentic AI Project

## Python Environment

### Virtual Environment Activation

The project uses a Python virtual environment (`.venv`) for dependency isolation.

**Activate the environment before running any Python commands:**
```powershell
# On Windows PowerShell
.venv\Scripts\Activate.ps1

# If execution policy blocks the script:
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".venv\Scripts\Activate.ps1")
```

When activated, the terminal prompt will show `(.venv)` prefix.

**All Python commands (python, pip, jupyter) must run within the activated environment.**

### Running Python Code

Use the activated environment to run scripts:
```powershell
python main.py
python -m <module_name>
```

For Jupyter notebooks:
```powershell
jupyter notebook
```

## Environment Configuration (.env File)

### .env File Location and Format
The `.env` file in the project root contains API keys and configuration:
```
_GROQ_API_KEY = "..."
GOOGLE_API_KEY = "..."
OPEN_API_KEY = "..."
```

### Loading Environment Variables
The project uses `python-dotenv` to load variables. Import it in Python code:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
```

### Development Practices
- **Never commit .env to version control** — .env should be in `.gitignore`
- **Do not expose API keys in code or logs**
- **Verify env file is loaded before using API keys** — check `os.getenv()` returns expected values
- **Keep .env keys consistent** with code that references them (e.g., `OPEN_API_KEY` used by main.py)

## Dependencies

### Key Libraries
- **langchain**: LLM framework and chains
- **langgraph**: Graph-based agent orchestration
- **python-dotenv**: Environment variable management
- **ipykernel**: Jupyter notebook kernel support

Install new dependencies using pip:
```powershell
pip install <package_name>
```

Update `requirements.txt` and `pyproject.toml` after installing packages:
```powershell
pip freeze > requirements.txt
```

## Running Commands

### Common Tasks

| Task | Command |
|------|---------|
| Run main script | `python main.py` |
| Start Jupyter | `jupyter notebook` |
| Install deps | `pip install -r requirements.txt` |
| Update deps | `pip freeze > requirements.txt` |
| Check Python version | `python --version` |

## Project Structure

```
.
├── main.py                    # Entry point
├── langchainbasics/
│   └── langchainbasics.ipynb  # Learning notebook
├── pyproject.toml             # Project metadata & dependencies
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (DO NOT COMMIT)
└── .venv/                     # Virtual environment (ignored in git)
```

## Development Guidelines

1. **Always activate .venv** before terminal operations
2. **Use absolute imports** in multi-file projects
3. **Load .env at module startup** in main scripts
4. **Test API connections** after .env changes to verify keys load correctly
5. **Keep notebooks in langchainbasics/** for learning/exploration code
6. **Use main.py** for production application logic
