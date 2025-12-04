# Development Guidelines

## Development Workflow
- Create feature branches from `main`
- Submit pull requests back to `main` (no direct commits to `main`)
- All PRs require passing tests before merge
- Once merged, Auto-sync to arpanauts via GitHub Action

## Before Submitting a PR
Run all code quality checks locally:
```bash
# Check everything (ruff, black, pyright, pytests)
./scripts/check.sh

# Or auto-fix formatting/linting issues first
./scripts/fix.sh
./scripts/check.sh
```

**Note:** CI runs these same checks automatically on every PR. All checks must pass before merging.

## Commit Messages
We use [Conventional Commits](https://www.conventionalcommits.org/):

Type | Description | Example
-- | -- | --
feat: | New features | feat: add protein annotation support
fix: | Bug fixes | fix: resolve DataFrame corruption in Step 3
docs: | Documentation only | docs: update installation instructions
test: | Test additions/changes | test: add unit tests for harmonization pipeline
refactor: | Code restructuring (no behavior change) | refactor: extract validation logic to separate module
chore: | Dependency updates, tooling | chore: update pandas to 2.2.0

## Code Style
We use automated tools to maintain code quality:
- **Black** - Code formatting
- **Ruff** - Linting and import sorting  
- **Pyright** - Type checking

### Writing Code
- Use type hints
- Add docstrings for public functions and classes

Example:
```python
def harmonize_phenotype(raw_data: pd.DataFrame, schema: str) -> pd.DataFrame:
    """
    Harmonize phenotype data to target schema.
    
    Args:
        raw_data: Input DataFrame with raw phenotype measurements
        schema: Target harmonization schema name
        
    Returns:
        Harmonized DataFrame with standardized columns
    """
    ...
```

### IDE Setup (Optional)
For real-time checking in your IDE, install these extensions/plugins:
- **Ruff** - For inline linting warnings
- **Black** - For format-on-save (VS Code users: install extension from ms-python)
- **Pyright** (optional) - For real-time type checking

Exact installation steps vary by IDE.

## Project Tracking

* Use GitHub Issues for task tracking
* Move cards across Kanban board:
* * Backlog - Identified tasks
* * Ready - Tasks that are ready to be worked on
* * In progress - Tasks that are actively being worked on
* * In review - Tasks that are complete and pending PR
* * Done - Tasks that have had the relevant PR merged
* Link PRs to issues: Closes #42 in PR description
* Label issues: bug, enhancement, documentation, planning

## Dependency Management
### Adding New Packages
When adding dependencies, always commit both files:
```bash
uv add <package-name>
git add pyproject.toml uv.lock
git commit -m "chore: add <package-name> dependency"
```

## .gitignore Best Practices
Ensure your .gitignore includes:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# uv
.uv/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/
*.ipynb  # Optional: if notebooks shouldn't be tracked

# Data (common for biomedical projects)
data/raw/
data/processed/
*.csv
*.parquet
*.h5

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```
Note: If you need to track specific data files, use `!data/examples/*.csv` to override