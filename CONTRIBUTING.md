# Contributing to Transformers Examples

Thank you for your interest in contributing to this repository! This guide will help you get started.

## 🛠️ Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Transformers-Examples.git
   cd Transformers-Examples
   ```
3. Run the setup script:
   ```bash
   ./setup.sh --venv
   ```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Remove any hardcoded API keys or tokens

### Security
- **Never commit API keys or tokens** to the repository
- Use environment variables for sensitive data
- Follow the `.env.example` template

### Adding New Examples
1. Create examples in appropriate directories:
   - `Genel-X/` for general transformer examples
   - `Vision Transformers/` for vision-related examples
   - `Multi Modal/` for multimodal examples
   - etc.

2. Include proper documentation:
   - Add comments explaining the code
   - Update the main README.md if needed
   - Create a brief description of what the example demonstrates

3. Test your code:
   - Ensure your code runs without errors
   - Test with different configurations if applicable
   - Verify dependencies are listed in requirements.txt

### Pull Request Process
1. Create a feature branch from main
2. Make your changes
3. Test thoroughly
4. Update documentation as needed
5. Submit a pull request with:
   - Clear description of changes
   - Explanation of what the example demonstrates
   - Any new dependencies added

## 🐛 Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior

## 📝 Documentation

- Keep README.md updated
- Add comments to complex code sections
- Document any new dependencies
- Explain the purpose of new examples

## ✅ Code Quality

Before submitting:
- [ ] Code follows PEP 8 style guidelines
- [ ] No hardcoded secrets or API keys
- [ ] Code is properly commented
- [ ] Dependencies are documented
- [ ] Examples run without errors
- [ ] Documentation is updated

## 🤝 Community

- Be respectful and constructive
- Help others learn from your examples
- Share knowledge and best practices
- Welcome feedback and suggestions

## 📄 License

By contributing to this repository, you agree that your contributions will be licensed under the [MIT License](LICENSE).

Thank you for contributing! 🚀