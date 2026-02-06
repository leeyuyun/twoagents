import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, str]],
        on_chunk: Optional[Callable[[str], None]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        response = requests.post(url, json=payload, stream=True, timeout=self.timeout_s)
        using_openai = False
        if response.status_code == 404:
            url = f"{self.base_url}/v1/chat/completions"
            openai_payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if options:
                openai_payload.update(options)
            response = requests.post(
                url,
                json=openai_payload,
                stream=True,
                timeout=self.timeout_s,
            )
            using_openai = True
        response.raise_for_status()

        full_text = ""
        raw_lines: List[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if using_openai and line.startswith("data:"):
                line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
            raw_lines.append(line)
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in data:
                raise RuntimeError(data["error"])
            if using_openai:
                delta = data.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content", "")
            else:
                chunk = data.get("message", {}).get("content", "")
            if chunk:
                full_text += chunk
                if on_chunk:
                    on_chunk(chunk)
            if not using_openai and data.get("done"):
                break
        return full_text, raw_lines
