# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Counterfact, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email **security@counterfact.dev** with:

- A description of the vulnerability
- Steps to reproduce the issue
- The potential impact
- Any suggested fixes (if applicable)

We will acknowledge receipt within 48 hours and provide a more detailed response within 7 days.

## Scope

Security concerns for this project include:

- **API key exposure**: Counterfact handles LLM API keys via environment variables. We never log, cache, or transmit API keys.
- **LLM cache files**: The `.llm_cache.json` file stores LLM responses keyed by prompt hash. It may contain sensitive data from your pipeline. This file is gitignored by default.
- **Arbitrary code execution**: Counterfact re-executes user-defined pipeline functions during diagnostic runs. Only run diagnostics on pipelines you trust.

## Best Practices

- Never commit `.env` files or API keys to version control.
- Review `.llm_cache.json` before sharing — it may contain pipeline-specific data.
- Use environment variables (not hardcoded strings) for all API keys.
