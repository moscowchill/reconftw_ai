# ReconFTW-AI

Integrate LLMs with ReconFTW to interpret pentesting results by category. Supports both local models (via Ollama) and remote APIs including **Anthropic's Claude 4** and **OpenAI's GPT-4**.

## 🧠 What does it do?

It analyzes ReconFTW outputs (`osint/`, `subdomains/`, `hosts/`, `webs/`) and generates a report using your choice of LLM (local or remote), classifying the results based on the type of audience: executive, brief summary, or offensive bug bounty style. Prompts are loaded dynamically from a `prompts.json` file for easy customization.

## 🤖 Available Models

### Anthropic (Claude) - Best for security analysis:
- `claude-4-sonnet-20250522` (Recommended - Latest Claude 4, most capable)
- `claude-4-opus-20250522` (Most powerful Claude 4, but slower)
- `claude-3-5-haiku-20241022` (Fast and cost-effective)
- `claude-3-5-sonnet-20241022` (Previous generation, still very capable)

### OpenAI:
- `gpt-4` (Most capable)
- `gpt-4-turbo` (Faster GPT-4)
- `gpt-3.5-turbo` (Fast and cost-effective)

### Ollama (Local):
- `llama3:8b`
- `mistral:7b`
- `deepseek-r1:8b`
- `qwen2.5-coder:latest`
- Any other Ollama-supported model

## 📦 Installation

### Option 1: For Remote API Support (Claude 4, GPT-4)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key (choose one):
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# For OpenAI
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: For Local Ollama Support

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

2. Pull a model:
```bash
ollama pull llama3:8b  # or your preferred model
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the `prompts.json` file is present in the working directory (included in the repository) or provide a custom prompts file.

## 🧪 Usage

### Using Claude 4 (Recommended for best results):
```bash
python reconftw_ai.py \
  --results-dir /path/to/reconftw_results \
  --output-dir /path/to/output \
  --provider anthropic \
  --model claude-4-sonnet-20250522 \
  --output-format md \
  --report-type bughunter
```

### Using GPT-4:
```bash
python reconftw_ai.py \
  --results-dir /path/to/reconftw_results \
  --output-dir /path/to/output \
  --provider openai \
  --model gpt-4 \
  --output-format md \
  --report-type executive
```

### Using Local Ollama (Original):
```bash
python reconftw_ai.py \
  --results-dir /path/to/reconftw_results \
  --output-dir /path/to/output \
  --provider ollama \
  --model llama3:8b \
  --output-format md \
  --report-type brief
```

### Arguments:
- `--results-dir`: Input directory with `osint/`, `subdomains/`, `hosts/`, `webs/` (default: `./reconftw_output`)
- `--output-dir`: Where to save the report (default: `./reconftw_ai_output`)
- `--provider`: LLM provider: `ollama`, `anthropic`, or `openai` (default: `ollama`)
- `--model`: Model name to use (e.g. `llama3:8b`, `claude-4-sonnet-20250522`, `gpt-4`)
- `--api-key`: API key for remote providers (optional if using environment variables)
- `--output-format`: Output format: `txt` or `md` (default: `txt`)
- `--report-type`: Report style: `executive`, `brief`, or `bughunter` (default: `executive`)
- `--prompts-file`: JSON file containing prompt templates (default: `prompts.json`)

### Customizing Prompts
The `prompts.json` file defines the LLM prompts for each report type and category. You can modify it to tailor the output structure, tone, or focus. Example structure:
```json
{
  "executive": {
    "osint": "As a security analyst, create a 200-300 word executive summary...",
    ...
  },
  ...
}
```

## 🖥️ Minimum Hardware Requirements

### CPU-Only (Minimal Setup)
- **RAM**: 8 GB (for quantized 2B–7B models)
- **Processor**: 4-core or better
- **Storage**: 5–10 GB

### Recommended CPU Setup
- **RAM**: 16 GB or more (for LLaMA 3 8B / Mistral 7B)
- **Processor**: 8-core modern CPU
- **Storage**: 10–20 GB

### GPU Setup (Recommended for Speed)
- **RAM**: 8–16 GB system RAM
- **VRAM**:
  - 4 GB: Small models (Gemma 2B)
  - 6–8 GB: LLaMA 3 8B (quantized)
  - 12 GB+: LLaMA 13B
- **GPU**: NVIDIA GPU with CUDA (GTX 1060+)

### Notes
- Quantization (4-bit/8-bit) is highly recommended to save memory
- SSD recommended if ReconFTW output is large
- Works on Linux, macOS, and WSL

## ✅ Supported ReconFTW Categories
- `osint/`: leaks, credentials, GitHub, spoofing, etc.
- `subdomains/`: DNS, takeovers, bruteforce, cloud
- `hosts/`: IPs, ports, WAFs, vulnerabilities
- `webs/`: CMS, endpoints, JS, fuzzing, parameters
- `overview`: Global summary across all categories

## 📊 Report Types

### `--report-type executive`
> Tailored for CISOs, managers, and non-technical stakeholders. Provides 200-400 word summaries with 3-7 bullet points per category, focusing on business risks (e.g., financial, reputational).
```markdown
## SUBDOMAINS

**Summary**: The subdomain scan identified exposures that could lead to brand damage or data leaks.
- **Dangling DNS**: `admin-test.company.com` points to a non-existent S3 bucket, risking subdomain takeover.
- **CORS Misconfiguration**: `dev-api.company.com` allows any origin, potentially exposing sensitive data.
- **Impact**: Malicious actors could hijack assets or breach data.
- **Recommendation**: Remove unused DNS records, enforce strict CORS policies.
```

### `--report-type brief`
> A compact summary with exactly 5 bullet points per category, each 1-2 sentences, ranked by severity.
```markdown
## SUBDOMAINS

- **[1] S3 Takeover**: `admin-test.company.com` is vulnerable to takeover.
- **[2] CORS Misconfig**: `dev-api.company.com` allows `*` origins.
- **[3] Deprecated Subdomain**: Exposed outdated systems.
- **[4] Staging Exposure**: Unprotected staging environment detected.
- **[5] Recommendation**: Clean up DNS and monitor subdomains.
```

### `--report-type bughunter`
> Offensive-style output for pentesters or bug bounty hunters, with 300-500 word responses and 3-7 prioritized attack paths per category.
```markdown
## SUBDOMAINS

**Analysis**: The subdomain scan revealed exploitable misconfigurations.
- **Takeover**: `admin-test.company.com` (S3 bucket missing). Claim the bucket to host malicious content.
- **CORS**: `dev-api.company.com` allows `*`. Test for token leaks via `fetch()`.
- **Staging Endpoint**: Exposed admin interface; attempt auth bypass or XSS.
```

## 💰 Cost Protection Features

When using remote APIs (Anthropic/OpenAI), the tool includes several safeguards:

1. **File Size Limits**:
   - Individual files > 10MB are skipped
   - Total data per category limited to 50MB
   - Files are truncated at ~100KB of text

2. **Cost Estimation**:
   - Estimates API costs before processing
   - Shows total token count
   - Prompts for confirmation if cost > $1.00

3. **Data Warnings**:
   - Notifies when files are skipped due to size
   - Shows truncation messages in output

These limits prevent unexpectedly high API costs from large reconnaissance outputs while still providing comprehensive analysis.

## 🤝 Contributions
Pull requests and issues are welcome! To contribute new prompts, update the `prompts.json` file and test with various ReconFTW outputs.
