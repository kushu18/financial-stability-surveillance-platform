"""
Financial Contagion Monitor — src/reports/llm_client.py
M.S. Ramaiah Institute of Technology | AIP67 | 2026

OllamaClient — HTTP client for the local Ollama LLM server.
Generates prose sections of the crisis report from structured data.

RAM requirements:
  llama3.2:3b  →  6–8 GB RAM   (lightweight fallback for 8GB systems)
  llama3.1:8b  → 10–12 GB RAM  (recommended — best quality for RBI reports)

Ollama runs as a local system service on port 11434.
If unavailable, ReportGenerator falls back to template engine automatically.
"""

import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "600"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
OLLAMA_TIMEOUT   = 90  # seconds — generous for CPU inference


class OllamaUnavailableError(Exception):
    """Raised when Ollama server is not reachable."""
    pass


class OllamaClient:
    """
    Thin HTTP client for the Ollama /api/generate endpoint.

    Usage:
        client = OllamaClient()
        if client.is_available():
            text = client.generate(
                system="You are RBI's chief risk analyst.",
                user="Summarize the COVID crisis impact on Yes Bank."
            )
    """

    def __init__(self, model: str = None):
        self.model = model or OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Checks if the Ollama server is running and the model is available.
        Returns True/False — never raises.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check if our model (or a variant) is available
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """
        Calls Ollama /api/generate and returns the response text.

        Args:
            system:      System prompt (role + instructions)
            user:        User prompt (data + report request)
            max_tokens:  Max response tokens (default from .env)
            temperature: Sampling temperature (default from .env, 0.3 = factual)

        Returns:
            Generated text string

        Raises:
            OllamaUnavailableError: If server unreachable
            requests.RequestException: If HTTP error
        """
        payload = {
            "model":  self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {
                "num_predict": max_tokens or OLLAMA_MAX_TOKENS,
                "temperature": temperature or OLLAMA_TEMPERATURE,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
        except requests.ConnectionError:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except requests.Timeout:
            raise OllamaUnavailableError(
                f"Ollama timed out after {OLLAMA_TIMEOUT}s. "
                "Try a smaller model: ollama pull llama3.2:3b"
            )

        if resp.status_code != 200:
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
            return data.get("response", "").strip()
        except json.JSONDecodeError:
            raise OllamaUnavailableError("Ollama returned invalid JSON response.")

    def get_model_name(self) -> str:
        """Returns the model name being used."""
        return self.model

    def list_available_models(self) -> list:
        """Returns list of model names available in this Ollama installation."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []
