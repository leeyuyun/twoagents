import argparse

from .agents import build_agent_configs
from .ollama_client import OllamaClient
from .orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Ollama two-agent chat (zh-tw).")
    parser.add_argument("--model", default="qwen3:14b", help="Ollama model name.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--agent-a-model",
        default=None,
        help="Override model for Agent A (defaults to --model).",
    )
    parser.add_argument(
        "--agent-b-model",
        default=None,
        help="Override model for Agent B (defaults to --model).",
    )
    parser.add_argument(
        "--agent-a-base-url",
        default=None,
        help="Override base URL for Agent A (defaults to --base-url).",
    )
    parser.add_argument(
        "--agent-b-base-url",
        default=None,
        help="Override base URL for Agent B (defaults to --base-url).",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Initial discussion topic (if omitted, prompt on startup).",
    )
    parser.add_argument(
        "--agent-a-role",
        default=None,
        help="Role supplement for Agent A (if omitted, prompt on startup).",
    )
    parser.add_argument(
        "--agent-b-role",
        default=None,
        help="Role supplement for Agent B (if omitted, prompt on startup).",
    )
    parser.add_argument("--max-turns", type=int, default=40, help="Max turns.")
    parser.add_argument("--min-sat", type=int, default=95, help="Min satisfaction.")
    parser.add_argument(
        "--stable-rounds",
        type=int,
        default=2,
        help="Consecutive rounds required to stop.",
    )
    parser.add_argument(
        "--summary-keep-last",
        type=int,
        default=6,
        help="Keep the most recent N transcript entries; summarize the rest.",
    )
    parser.add_argument(
        "--summary-max-points",
        type=int,
        default=12,
        help="Max bullet points to keep in the summary.",
    )
    parser.add_argument(
        "--transcript-path",
        default=None,
        help="Optional transcript file path (JSONL).",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=120,
        help="HTTP read timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_topic = "人生的意義"
    try:
        topic = args.topic or input(
            f"請輸入討論主題 (default: {default_topic}): "
        ).strip()
    except EOFError:
        topic = args.topic or default_topic
    if not topic:
        topic = default_topic

    def read_role(prompt: str) -> str:
        try:
            value = input(prompt).strip()
        except EOFError:
            return ""
        return value

    agent_a_role = args.agent_a_role or read_role(
        "Agent A 角色補充 (default: 無): "
    )
    agent_b_role = args.agent_b_role or read_role(
        "Agent B 角色補充 (default: 無): "
    )
    agents = build_agent_configs(
        agent_a_role=agent_a_role or None,
        agent_b_role=agent_b_role or None,
    )
    agent_a_model = args.agent_a_model or args.model
    agent_b_model = args.agent_b_model or args.model
    agent_a_base_url = args.agent_a_base_url or args.base_url
    agent_b_base_url = args.agent_b_base_url or args.base_url
    clients = {
        "Agent A": OllamaClient(
            base_url=agent_a_base_url,
            model=agent_a_model,
            timeout_s=args.timeout_s,
        ),
        "Agent B": OllamaClient(
            base_url=agent_b_base_url,
            model=agent_b_model,
            timeout_s=args.timeout_s,
        ),
    }
    orchestrator = Orchestrator(
        clients=clients,
        agents=agents,
        max_turns=args.max_turns,
        min_satisfaction=args.min_sat,
        stable_rounds=args.stable_rounds,
        summary_keep_last=args.summary_keep_last,
        summary_max_points=args.summary_max_points,
        initial_user_prompt=f"請開始對話，主題：{topic}。",
        topic=topic,
        transcript_path=args.transcript_path,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
