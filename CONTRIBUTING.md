# Contributing to VeriTriage

Thank you for your interest in contributing to VeriTriage! This document provides guidelines and instructions for contributing.

## How to Contribute

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/veritriage.git
cd veritriage
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Make Changes
- Create or modify files in appropriate directories
- Follow Python PEP 8 style guidelines
- Add comments and docstrings

### 5. Test Your Changes
```bash
# Run Jupyter notebooks to verify results
jupyter notebook notebooks/03_model_training.ipynb
```

### 6. Commit & Push
```bash
git add .
git commit -m "feat: description of your changes"
git push origin feature/your-feature-name
```

### 7. Create Pull Request
- Open a PR on GitHub with a clear description
- Reference any related issues

## Code Guidelines

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Notebooks**: Keep cells focused and well-documented
- **Functions**: Include docstrings with parameter descriptions
- **Variable Names**: Use descriptive, snake_case names

## Areas for Contribution

- 🐛 Bug fixes
- ✨ Feature improvements
- 📊 Analysis enhancements
- 📚 Documentation
- 🧪 Additional test notebooks
- 🔧 Pipeline optimization

## Reporting Issues

Found a bug? Please open a GitHub issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- System information

## Questions?

Open a GitHub discussion or email tushar.dudeja@gmail.com

---

Thank you for contributing to VeriTriage! 🚀
