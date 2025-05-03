ReconFTW-AI
Integrate a local LLM with ReconFTW to interpret pentesting results by category (tested with mistral:7b, llama3:8b, deepseek-r1:8b, and qwen2.5-coder:latest).
🧠 What does it do?
It analyzes ReconFTW outputs (osint/, subdomains/, hosts/, webs/) and generates a report using a local LLM, classifying the results based on the type of audience: executive, brief summary, or offensive bug bounty style. Prompts are loaded dynamically from a prompts.json file for easy customization.
📦 Installation

Install Ollama:

curl https://ollama.ai/install.sh | sh


Pull a model:

ollama pull llama3:8b  # or your preferred model


Install dependencies:

pip install -r requirements.txt


Ensure the prompts.json file is present in the working directory (included in the repository) or provide a custom prompts file.

🧪 Usage
Basic usage:
python reconftw_ai.py \
  --results-dir /path/to/reconftw_results \
  --output-dir /path/to/output \
  --model llama3:8b \
  --output-format md \
  --report-type bughunter \
  --prompts-file prompts.json

Arguments:

--results-dir: Input directory with osint/, subdomains/, hosts/, webs/ (default: ./reconftw_output)
--output-dir: Where to save the report (default: ./reconftw_ai_output)
--model: Ollama model to use (default: llama3)
--output-format: Output format: txt or md (default: txt)
--report-type: Report style: executive, brief, or bughunter (default: executive)
--prompts-file: JSON file containing prompt templates (default: prompts.json)

Customizing Prompts
The prompts.json file defines the LLM prompts for each report type and category. You can modify it to tailor the output structure, tone, or focus. Example structure:
{
  "executive": {
    "osint": "As a security analyst, create a 200-300 word executive summary...",
    ...
  },
  ...
}

🖥️ Minimum Hardware Requirements
CPU-Only (Minimal Setup)

RAM: 8 GB (for quantized 2B–7B models)
Processor: 4-core or better
Storage: 5–10 GB

Recommended CPU Setup

RAM: 16 GB or more (for LLaMA 3 8B / Mistral 7B)
Processor: 8-core modern CPU
Storage: 10–20 GB

GPU Setup (Recommended for Speed)

RAM: 8–16 GB system RAM
VRAM:
4 GB: Small models (Gemma 2B)
6–8 GB: LLaMA 3 8B (quantized)
12 GB+: LLaMA 13B


GPU: NVIDIA GPU with CUDA (GTX 1060+)

Notes

Quantization (4-bit/8-bit) is highly recommended to save memory
SSD recommended if ReconFTW output is large
Works on Linux, macOS, and WSL

✅ Supported ReconFTW Categories

osint/: leaks, credentials, GitHub, spoofing, etc.
subdomains/: DNS, takeovers, bruteforce, cloud
hosts/: IPs, ports, WAFs, vulnerabilities
webs/: CMS, endpoints, JS, fuzzing, parameters
overview: Global summary across all categories

📊 Report Types
--report-type executive

Tailored for CISOs, managers, and non-technical stakeholders. Provides 200-400 word summaries with 3-7 bullet points per category, focusing on business risks (e.g., financial, reputational).

## SUBDOMAINS

**Summary**: The subdomain scan identified exposures that could lead to brand damage or data leaks.
- **Dangling DNS**: `admin-test.company.com` points to a non-existent S3 bucket, risking subdomain takeover.
- **CORS Misconfiguration**: `dev-api.company.com` allows any origin, potentially exposing sensitive data.
- **Impact**: Malicious actors could hijack assets or breach data.
- **Recommendation**: Remove unused DNS records, enforce strict CORS policies.

--report-type brief

A compact summary with exactly 5 bullet points per category, each 1-2 sentences, ranked by severity.

## SUBDOMAINS

- **[1] S3 Takeover**: `admin-test.company.com` is vulnerable to takeover.
- **[2] CORS Misconfig**: `dev-api.company.com` allows `*` origins.
- **[3] Deprecated Subdomain**: Exposed outdated systems.
- **[4] Staging Exposure**: Unprotected staging environment detected.
- **[5] Recommendation**: Clean up DNS and monitor subdomains.

--report-type bughunter

Offensive-style output for pentesters or bug bounty hunters, with 300-500 word responses and 3-7 prioritized attack paths per category.

## SUBDOMAINS

**Analysis**: The subdomain scan revealed exploitable misconfigurations.
- **Takeover**: `admin-test.company.com` (S3 bucket missing). Claim the bucket to host malicious content.
- **CORS**: `dev-api.company.com` allows `*`. Test for token leaks via `fetch()`.
- **Staging Endpoint**: Exposed admin interface; attempt auth bypass or XSS.

🤝 Contributions
Pull requests and issues are welcome! To contribute new prompts, update the prompts.json file and test with various ReconFTW outputs.
