# Security Policy

## Reporting a Vulnerability

We take the security of the LLM Fine-Tuning Pipeline seriously. If you discover
a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please email **[security@example.com](mailto:security@example.com)** with:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Impact assessment** — what could an attacker achieve?
4. **Suggested fix** (if any)

### Response Timeline

| Action | Timeline |
|--------|----------|
| Acknowledgment of report | Within 48 hours |
| Initial assessment | Within 5 business days |
| Patch development | Within 30 days (critical), 90 days (non-critical) |
| Public disclosure | After patch is released |


## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅ Active support |
| < 1.0   | ❌ No longer supported |



## Security Best Practices

### Secrets Management

This project uses environment variables for all sensitive credentials.
**Never** commit secrets to version control.

| Secret | Environment Variable | Purpose |
|--------|---------------------|---------|
| Hugging Face Token | `HF_TOKEN` | Model/dataset access |
| Weights & Biases Key | `WANDB_API_KEY` | Experiment tracking |
| OpenAI API Key | `OPENAI_API_KEY` | LLM-as-judge evaluation |

#### Setup

```bash
# Copy the template
cp .env.example .env

# Edit with your credentials
# NEVER commit .env to version control
```

The `.gitignore` file is configured to exclude `.env` and other secret files.

### Docker Security

- The production Dockerfile runs as a **non-root user** (`appuser`)
- Base images are pinned to specific versions
- Only necessary files are copied into the container
- Health checks are configured for all long-running services

### Dependency Security

- All dependencies are **version-pinned** in `requirements.txt` and `pyproject.toml`
- GitHub Actions runs automated **dependency review** on pull requests
- **CodeQL** analysis scans for code vulnerabilities weekly
- Run `pip-audit` locally to check for known vulnerabilities:
  ```bash
  pip install pip-audit
  pip-audit
  ```

### Model Security

- Model weights are loaded from trusted sources (Hugging Face Hub)
- Use `safetensors` format (default) instead of pickle-based formats
- Verify model checksums when downloading gated models
- Never execute untrusted model code — use `trust_remote_code=False` (default)

### Data Security

- Training data is stored locally in `data/` (gitignored)
- No PII should be included in training datasets
- Validate and sanitize all input data through the data pipeline
- Export formats (JSONL, Parquet, Arrow) do not execute code

### Network Security

- The inference server binds to `0.0.0.0` only inside Docker containers
- CORS is configured with explicit allowed origins
- Rate limiting is applied via Prometheus metrics monitoring
- Use HTTPS in production deployments behind a reverse proxy

## Known Security Considerations

1. **BitsAndBytes**: The `bitsandbytes` library loads CUDA kernels at runtime.
   Ensure you trust the source and version.

2. **Remote Code Execution**: Some Hugging Face models require
   `trust_remote_code=True`. This project defaults to `False`. Only enable
   this for models you explicitly trust.

3. **Inference API**: The FastAPI inference server does not include
   authentication by default. Add authentication middleware for production
   deployments.



## Security Updates

Security patches are released as soon as possible after a vulnerability is
confirmed. Subscribe to repository releases to receive notifications.
