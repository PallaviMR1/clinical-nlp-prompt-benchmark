"""
Local LLM Client for Ollama.
Manages HTTP requests, latency measurement, token usage extraction,
dynamic model inspection, and robust error handling.
"""

import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"

class LocalLLMClient:
    """
    Client for local Ollama instance with execution telemetry
    and live model architecture inspection.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def check_health(self) -> bool:
        """Verify if Ollama service is reachable and default/specified model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Direct match or substring match
                    return any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
            return False
        except Exception:
            return False

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Returns full health status dictionary including server status,
        active model details, and all locally downloaded models.
        """
        status: Dict[str, Any] = {
            "online": False,
            "base_url": self.base_url,
            "active_model": self.model,
            "available_models": [],
            "model_details": {},
            "error": None
        }

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    status["online"] = True
                    data = json.loads(response.read().decode("utf-8"))
                    raw_models = data.get("models", [])
                    status["available_models"] = [m.get("name", "") for m in raw_models]
                    
                    # Fetch details for the active model
                    status["model_details"] = self.get_model_details(self.model)
        except Exception as e:
            status["error"] = str(e)

        return status

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns list of models installed in local Ollama with summary details."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    return data.get("models", [])
            return []
        except Exception:
            return []

    def get_model_details(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Queries Ollama /api/show to inspect live architecture specifications:
        parameter size, quantization level, context window, layer count, and GQA heads.
        """
        target_model = model_name or self.model
        details_info: Dict[str, Any] = {
            "name": target_model,
            "parameter_size": "1.54B",
            "quantization": "GGUF Q4_K_M",
            "family": "Qwen2",
            "context_window": 32768,
            "layers": 28,
            "attention_heads": 12,
            "kv_heads": 2,
            "hidden_dim": 1536,
            "vocab_size": 151643,
            "architecture": "Autoregressive Decoder-only Transformer (GQA + SwiGLU + RoPE)"
        }

        try:
            payload = json.dumps({"name": target_model}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=6) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    raw_det = data.get("details", {})
                    mi = data.get("model_info", {})

                    if raw_det.get("parameter_size"):
                        details_info["parameter_size"] = raw_det.get("parameter_size")
                    if raw_det.get("quantization_level"):
                        details_info["quantization"] = raw_det.get("quantization_level")
                    if raw_det.get("family"):
                        details_info["family"] = str(raw_det.get("family")).capitalize()

                    # Scan model_info for architecture keys
                    for k, v in mi.items():
                        if "context_length" in k:
                            details_info["context_window"] = v
                        elif "block_count" in k:
                            details_info["layers"] = v
                        elif "attention.head_count_kv" in k:
                            details_info["kv_heads"] = v
                        elif "attention.head_count" in k:
                            details_info["attention_heads"] = v
                        elif "embedding_length" in k:
                            details_info["hidden_dim"] = v
                        elif "vocab_size" in k:
                            details_info["vocab_size"] = v
                        elif "architecture" in k:
                            details_info["architecture"] = f"{v} Transformer"
        except Exception:
            # Fall back to known defaults for Qwen2.5:1.5b
            pass

        return details_info

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        enforce_json_format: bool = False,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Sends generation request to local Ollama instance.
        Returns a dictionary with raw response text, latency, token metrics, and metadata.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "top_p": 0.9,
                "num_predict": 1024
            }
        }
        if enforce_json_format:
            payload["format"] = "json"

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )

        t_start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.status
                raw_bytes = response.read()
            t_end = time.perf_counter()
            wall_clock_time = t_end - t_start

            if status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP status {status_code}",
                    "response": "",
                    "latency_sec": round(wall_clock_time, 3),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_sec": 0.0,
                    "model": self.model
                }

            res_json = json.loads(raw_bytes.decode("utf-8"))
            raw_text = res_json.get("response", "")
            prompt_eval_count = res_json.get("prompt_eval_count", 0)
            eval_count = res_json.get("eval_count", 0)
            eval_duration_ns = res_json.get("eval_duration", 0)
            tokens_per_sec = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

            return {
                "success": True,
                "response": raw_text,
                "latency_sec": round(wall_clock_time, 3),
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval_count + eval_count,
                "tokens_per_sec": round(tokens_per_sec, 2),
                "model": self.model
            }

        except urllib.error.URLError as e:
            t_end = time.perf_counter()
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "response": "",
                "latency_sec": round(t_end - t_start, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens_per_sec": 0.0,
                "model": self.model
            }
        except Exception as e:
            t_end = time.perf_counter()
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response": "",
                "latency_sec": round(t_end - t_start, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens_per_sec": 0.0,
                "model": self.model
            }

if __name__ == "__main__":
    client = LocalLLMClient()
    print("Health check:", client.check_health())
    details = client.get_model_details()
    print("Model Details:", details)
