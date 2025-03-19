# Cursor Rules

Cursor rules are configuration files that help guide AI-assisted development in the Cursor IDE. They provide structured instructions for how the AI should behave in specific contexts or when working with certain types of files.

## What is Cursor?

[Cursor](https://www.cursor.com/) is an AI-powered IDE that helps developers write, understand, and maintain code more efficiently. It integrates AI capabilities directly into the development workflow, providing features like:

- AI-assisted code completion
- Natural language code generation
- Intelligent code explanations
- Automated refactoring suggestions

## Understanding Cursor Rules

Cursor rules are defined in `.mdc` files within the `.cursor/rules` directory. Each rule file contains:

1. **Metadata Header**: YAML frontmatter that defines:
   - `description`: When the rule should be applied
   - `globs`: File patterns the rule applies to
   - `alwaysApply`: Whether the rule should be applied automatically

2. **Rule Content**: Markdown-formatted instructions that guide the AI's behavior

## Available Rules

Currently, the following rules are defined:

### `simple-language.mdc`
- **Purpose**: Ensures documentation is written at a grade 10 reading level
- **Applies to**: Markdown files (*.md)
- **Auto Apply**: No
- **Key Requirements**: 
  - Write at grade 10 reading level
  - Ensure code blocks are self-contained with complete imports

### `new-features-planning.mdc`
- **Purpose**: Guides feature implementation workflow
- **Applies to**: Python files (*.py)
- **Auto Apply**: Yes
- **Key Requirements**:
  - Create new branch from main
  - Make incremental commits
  - Create todo.md for large features
  - Start pull requests using gh
  - Include "This PR was written by [Cursor](cursor.com)" in PRs

### `followups.mdc`
- **Purpose**: Ensures thoughtful follow-up suggestions
- **Applies to**: All files
- **Auto Apply**: Yes
- **Key Requirements**:
  - Generate actionable hotkey suggestions using [J], [K], [L]
  - Focus on small, contextual code changes
  - Suggestions should be thoughtful and actionable

## Creating New Rules

To create a new rule:

1. Create a `.mdc` file in `.cursor/rules/`
2. Add YAML frontmatter with required metadata:
   ```yaml
   ---
   description: when to apply this rule
   globs: file patterns to match
   alwaysApply: true/false
   ---
   ```
3. Write clear, specific instructions in Markdown
4. Test the rule with relevant file types

## Best Practices

- Keep rules focused and specific
- Use clear, actionable language
- Test rules thoroughly before committing
- Document any special requirements or dependencies
- Update rules as project needs evolve
