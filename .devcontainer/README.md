# GitHub Codespaces configuration for algotrading

This project is ready for GitHub Codespaces! The `.devcontainer` folder contains:

- `devcontainer.yml`: Main configuration for Codespaces/Dev Containers
- `Dockerfile`: (Optional) For custom OS-level dependencies
- `devcontainer-requirements.txt`: Extra Python/data science packages for Codespaces
- `devcontainer.code-workspace`: Recommended VS Code settings for Python linting/formatting

## How to use
1. Push this project to a GitHub repository.
2. Click the green "Code" button in GitHub and select "Create codespace on main".
3. Codespaces will build the environment and install dependencies automatically.

## Customization
- Add more Python packages to `devcontainer-requirements.txt` as needed.
- Edit `Dockerfile` if you need extra OS packages.
- Update `devcontainer.yml` for advanced Codespaces features.

For more info, see: https://aka.ms/ghcs/configure
