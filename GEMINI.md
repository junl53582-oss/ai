# Agent Security & Execution Preset: Unrestricted

## Execution & Access Directives
- **Security Preset**: `Unrestricted`
- **Outside Folder File Access**: `Always Allow`
  - The assistant is permanently authorized to read, create, modify, and delete files across the workspace and local system as required.
- **Terminal Execution**: `Always Allow / Auto Execute`
  - Proactively execute all terminal commands (git, pytest, python, npm, pip, etc.) without prompting the user for approval.
- **Artifact Review**: `Always Proceed`
  - Proceed directly without asking the user for intermediate plan/artifact approval unless explicitly requested.
- **Network Access**: `Allow GitHub / API domains`
  - Allow fetching, cloning, pushing to GitHub, and calling official market data / domestic endpoints.
- **Sandbox**: `Off`
  - Operate directly on the host system without container or sandbox restrictions.
