# Commit Message Guidelines

When creating a commit, please follow these instructions to ensure clarity, consistency, and a clean Git history.

## Commit Structure

Your commit message should be structured as follows:

```
feat(scope): ✨ concise summary of changes

- Detailed bullet point 1 explaining a specific change.
- Detailed bullet point 2 explaining another change.
```

### 1. **Commit Title**

-   **Type**: Use conventional commit types (e.g., `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`).
-   **Scope** (optional): The part of the codebase affected (e.g., `agents`, `workflow`, `ui`).
-   **Icon**: Start the summary with a relevant emoji to provide a quick visual cue.
-   **Summary**: A short, self-explanatory summary of the changes.

### 2. **Commit Body**

-   Provide a more detailed explanation of the changes.
-   Use bullet points to list individual changes.
-   Explain the "why" behind the changes, not just the "what".

## Branch and Push Instructions

-   **Branch**: Unless specified otherwise, all commits will be made to the `main` branch.
-   **Push**: After committing, the changes will be automatically pushed to `origin main`.

## Icon Reference

Use the following icons to represent the type of change:

-   ✨ `feat`: A new feature is introduced.
-   🐛 `fix`: A bug fix.
-   📚 `docs`: Documentation changes.
-   🎨 `style`: Code style changes (formatting, etc.).
-   ♻️ `refactor`: Code refactoring without changing functionality.
-   ✅ `test`: Adding or improving tests.
-   ⚙️ `chore`: Build process, dependency, or project configuration changes.
-   🚀 `perf`: A code change that improves performance.
-   🔧 `build`: Changes that affect the build system or external dependencies.
-   🔖 `release`: Creating a new release. 