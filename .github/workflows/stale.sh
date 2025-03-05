#!/bin/bash
# Install GitHub CLI first: https://cli.github.com/

# Get all open issues created before 2025
gh issue list --repo instructor-ai/instructor --state open --json number,title,createdAt --limit 1000 | \
jq '.[] | select(.createdAt < "2025-01-01")' | \
jq -r '.number' | \
while read -r issue_number; do
  echo "Closing issue #$issue_number"
  gh issue close "$issue_number" --repo instructor-ai/instructor --comment "Closing as part of repository maintenance for issues created before 2025."
done

# Get all open pull requests created before 2025
gh pr list --repo instructor-ai/instructor --state open --json number,title,createdAt --limit 1000 | \
jq '.[] | select(.createdAt < "2025-01-01")' | \
jq -r '.number' | \
while read -r pr_number; do
  echo "Closing pull request #$pr_number"
  gh pr close "$pr_number" --repo instructor-ai/instructor --comment "Closing as part of repository maintenance for pull requests created before 2025."
done