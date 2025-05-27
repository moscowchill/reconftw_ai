#!/usr/bin/env python3

import os
import argparse
import glob
import sys
import subprocess
import shutil
import json
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Import for remote API support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Default configuration
DEFAULT_RECONFTW_RESULTS_DIR = "./reconftw_output"
DEFAULT_OUTPUT_DIR = "./reconftw_ai_output"
DEFAULT_MODEL_NAME = "llama3"
DEFAULT_OUTPUT_FORMAT = "txt"
DEFAULT_REPORT_TYPE = "executive"
DEFAULT_PROMPTS_FILE = "prompts.json"
DEFAULT_PROVIDER = "ollama"

REPORT_TYPES = ["executive", "brief", "bughunter"]
OUTPUT_FORMATS = ["txt", "md"]
CATEGORIES = ["osint", "subdomains", "hosts", "webs"]
PROVIDERS = ["ollama", "anthropic", "openai"]

# Model mappings for different providers
ANTHROPIC_MODELS = ["claude-4-opus-20250522", "claude-4-sonnet-20250522", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]
OPENAI_MODELS = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    """Ollama provider for local models"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama package not installed. Run: pip install ollama")
        ensure_ollama_running()
        ensure_model_available(model_name)
    
    def generate(self, prompt: str) -> str:
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response.get("response", "[Error] Empty response from model.")
        except Exception as e:
            return f"[Error] Failed to generate with Ollama: {str(e)}"

class AnthropicProvider(LLMProvider):
    """Anthropic provider for Claude models"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or use --api-key")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"[Error] Failed to generate with Anthropic: {str(e)}"

class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models"""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Get API key from environment or parameter
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Error] Failed to generate with OpenAI: {str(e)}"

def get_provider(provider_name: str, model_name: str, api_key: Optional[str] = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider"""
    if provider_name == "ollama":
        return OllamaProvider(model_name, api_key)
    elif provider_name == "anthropic":
        return AnthropicProvider(model_name, api_key)
    elif provider_name == "openai":
        return OpenAIProvider(model_name, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def load_prompts(prompts_file: str) -> Dict:
    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Prompts file '{prompts_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in prompts file: {e}")
        sys.exit(1)

def ensure_ollama_running():
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print("[!] Ollama is not running. Attempting to start it in background...")
        if shutil.which("ollama") is None:
            print("[ERROR] Ollama not installed or not in PATH.")
            sys.exit(1)
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ensure_model_available(model_name):
    try:
        output = subprocess.check_output(["ollama", "list"], encoding="utf-8")
        if model_name not in output:
            print(f"[*] Model '{model_name}' not found. Downloading with 'ollama pull {model_name}'...")
            subprocess.run(["ollama", "pull", model_name], check=True)
    except Exception as e:
        print(f"[ERROR] Could not verify or pull model: {e}")
        sys.exit(1)

def read_files(category: str, results_dir: str) -> str:
    combined_data = ""
    category_dir = os.path.join(results_dir, category)

    if not os.path.isdir(category_dir):
        return f"[Error] Directory {category_dir} does not exist."

    file_paths = glob.glob(os.path.join(category_dir, "**/*"), recursive=True)

    for file_path in file_paths:
        if os.path.isfile(file_path):
            relative_path = os.path.relpath(file_path, results_dir)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_data += f"--- {relative_path} ---\n{f.read().strip()}\n"
            except Exception as e:
                combined_data += f"[Error] Failed to read {relative_path}: {str(e)}\n"

    if not combined_data:
        return f"[Info] No files found in {category_dir}."

    return combined_data.strip()

def process_category(category: str, data: str, provider: LLMProvider, report_type: str, base_prompts: Dict) -> str:
    if not data:
        return f"[Error] No data available for {category}."

    prompt_template = base_prompts.get(report_type, {}).get(category, "Analyze this data:\n{data}")
    prompt = prompt_template.format(data=data)

    return provider.generate(prompt)

def save_results(results: Dict[str, str], output_dir: str, model_name: str, provider_name: str, output_format: str, report_type: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "md" if output_format == "md" else "txt"
    output_file = os.path.join(output_dir, f"reconftw_analysis_{report_type}_{timestamp}.{extension}")

    with open(output_file, "w", encoding="utf-8") as f:
        if output_format == "md":
            f.write(f"# ReconFTW-AI Analysis\n\n")
            f.write(f"- **Model Used**: `{model_name}` (Provider: {provider_name})\n")
            f.write(f"- **Report Type**: `{report_type}`\n")
            f.write(f"- **Date**: `{timestamp}`\n\n")
            for category, interpretation in results.items():
                f.write(f"## {category.upper()}\n\n{interpretation}\n\n")
        else:
            f.write(f"ReconFTW-AI Analysis\nModel: {model_name} (Provider: {provider_name})\nReport Type: {report_type}\nDate: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            for category, interpretation in results.items():
                f.write(f"=== {category.upper()} ===\n{interpretation}\n\n")

    print(f"[*] Results saved to '{output_file}'")

def analyze_reconftw_results(results_dir: str, provider: LLMProvider, report_type: str, base_prompts: Dict) -> Dict[str, str]:
    results = {}
    all_data = ""

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(read_files, category, results_dir): category
            for category in CATEGORIES
        }

        raw_data_per_category = {}
        for future in as_completed(futures):
            category = futures[future]
            raw_data = future.result()
            raw_data_per_category[category] = raw_data

    # Process categories sequentially to avoid rate limits
    for category in CATEGORIES:
        print(f"[*] Processing {category}...")
        interpretation = process_category(category, raw_data_per_category[category], provider, report_type, base_prompts)
        results[category] = interpretation
        all_data += f"{category.upper()}:\n{raw_data_per_category[category]}\n\n"

    print("[*] Processing overview...")
    results["overview"] = process_category("overview", all_data, provider, report_type, base_prompts)
    return results

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> bool:
    if not os.path.isdir(args.results_dir):
        print(f"[Error] Results directory '{args.results_dir}' does not exist.")
        parser.print_help()
        return False
    if args.output_format not in OUTPUT_FORMATS:
        print(f"[Error] Invalid format '{args.output_format}'. Choose from: {', '.join(OUTPUT_FORMATS)}")
        return False
    if args.report_type not in REPORT_TYPES:
        print(f"[Error] Invalid report type '{args.report_type}'. Choose from: {', '.join(REPORT_TYPES)}")
        return False
    if args.provider not in PROVIDERS:
        print(f"[Error] Invalid provider '{args.provider}'. Choose from: {', '.join(PROVIDERS)}")
        return False
    
    # Validate model names for specific providers
    if args.provider == "anthropic" and args.model not in ANTHROPIC_MODELS:
        print(f"[Warning] Model '{args.model}' may not be valid for Anthropic. Available models: {', '.join(ANTHROPIC_MODELS)}")
    elif args.provider == "openai" and args.model not in OPENAI_MODELS:
        print(f"[Warning] Model '{args.model}' may not be valid for OpenAI. Available models: {', '.join(OPENAI_MODELS)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="ReconFTW-AI: Use LLMs to interpret ReconFTW results",
        epilog="""
Examples:
  # Using local Ollama model:
  python reconftw_ai.py --results-dir ./reconftw_output --model llama3:8b --report-type executive

  # Using Claude 4 (requires ANTHROPIC_API_KEY environment variable):
  python reconftw_ai.py --results-dir ./reconftw_output --provider anthropic --model claude-4-sonnet-20250522 --report-type bughunter

  # Using GPT-4 with API key:
  python reconftw_ai.py --results-dir ./reconftw_output --provider openai --model gpt-4 --api-key "sk-..." --output-format md

  # Custom output directory and prompts file:
  python reconftw_ai.py --results-dir ./scan_results --output-dir ./reports --prompts-file custom_prompts.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default=DEFAULT_RECONFTW_RESULTS_DIR, help="Directory with ReconFTW results.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to save the analysis.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name to use (e.g. llama3, claude-4-sonnet-20250522, gpt-4).")
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER, help="LLM provider to use.")
    parser.add_argument("--api-key", help="API key for remote providers (can also use environment variables).")
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default=DEFAULT_OUTPUT_FORMAT, help="Output format: txt or md.")
    parser.add_argument("--report-type", choices=REPORT_TYPES, default=DEFAULT_REPORT_TYPE, help="Type of report to generate.")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE, help="JSON file containing prompt templates.")

    args = parser.parse_args()

    if not validate_args(args, parser):
        sys.exit(1)

    base_prompts = load_prompts(args.prompts_file)

    # Get the appropriate provider
    try:
        provider = get_provider(args.provider, args.model, args.api_key)
    except Exception as e:
        print(f"[ERROR] Failed to initialize provider: {e}")
        sys.exit(1)

    print(f"[*] Analyzing with model '{args.model}' via {args.provider} using report type '{args.report_type}'...")
    results = analyze_reconftw_results(args.results_dir, provider, args.report_type, base_prompts)

    for category, content in results.items():
        print(f"\n=== {category.upper()} ===\n{content[:500]}{'...' if len(content) > 500 else ''}")

    save_results(results, args.output_dir, args.model, args.provider, args.output_format, args.report_type)

if __name__ == "__main__":
    main()
