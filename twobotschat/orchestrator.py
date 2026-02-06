import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .agents import AgentConfig, build_messages
from .ollama_client import OllamaClient


@dataclass
class ParseResult:
    data: Optional[Dict]
    error: Optional[str]


class Orchestrator:
    def __init__(
        self,
        clients: Dict[str, OllamaClient],
        agents: Dict[str, AgentConfig],
        max_turns: int,
        min_satisfaction: int,
        stable_rounds: int,
        summary_keep_last: int,
        summary_max_points: int,
        initial_user_prompt: str,
        topic: str,
        transcript_path: Optional[str],
    ) -> None:
        self.clients = clients
        self.agents = agents
        self.max_turns = max_turns
        self.min_satisfaction = min_satisfaction
        self.stable_rounds = stable_rounds
        self.summary_keep_last = summary_keep_last
        self.summary_max_points = summary_max_points
        self.initial_user_prompt = initial_user_prompt
        self.topic = topic
        self.transcript_path = transcript_path
        self.transcript: List[Dict] = []
        self.context: List[Dict] = []

    def run(self) -> None:
        stable_count = 0
        stop_reason = ""
        last_valid: Dict[str, Dict] = {}

        for turn in range(1, self.max_turns + 1):
            print(f"\n=== Turn {turn} ===")
            for agent_name in ["Agent A", "Agent B"]:
                agent = self.agents[agent_name]
                parsed_entry = self._run_agent_turn(
                    turn=turn,
                    agent=agent,
                    initial_user_prompt=self.initial_user_prompt,
                )
                if parsed_entry and not parsed_entry.get("parse_error"):
                    last_valid[agent_name] = parsed_entry

            latest_a = self._latest_entry("Agent A")
            latest_b = self._latest_entry("Agent B")
            sat_a = self._get_satisfaction(latest_a)
            sat_b = self._get_satisfaction(latest_b)

            if sat_a is not None and sat_b is not None:
                if sat_a >= self.min_satisfaction and sat_b >= self.min_satisfaction:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0

            if stable_count >= self.stable_rounds:
                stop_reason = (
                    f"連續 {self.stable_rounds} 輪雙方滿意度皆 >= {self.min_satisfaction}。"
                )
                break

        if not stop_reason:
            stop_reason = (
                f"達到 max_turns = {self.max_turns} 仍未達成終止條件。"
            )

        transcript_path = self._write_transcript()
        self._print_final_report(
            last_valid=last_valid,
            stop_reason=stop_reason,
            reached_target=stable_count >= self.stable_rounds,
        )
        print(f"\nTranscript: {transcript_path}")

    def _run_agent_turn(
        self,
        turn: int,
        agent: AgentConfig,
        initial_user_prompt: str,
    ) -> Optional[Dict]:
        for attempt in [1, 2]:
            strict_json = attempt == 2
            summary, recent = self._summarize_and_trim_context()
            messages = build_messages(
                agent=agent,
                transcript=recent,
                initial_user_prompt=initial_user_prompt,
                strict_json=strict_json,
                summary=summary,
                topic=self.topic,
            )

            print(f"{agent.name}:", end=" ", flush=True)
            client = self.clients[agent.name]
            output, raw_lines = client.chat(
                messages=messages,
                on_chunk=lambda chunk: print(chunk, end="", flush=True),
            )
            print("")

            parse_result = self._parse_output(output)
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "turn": turn,
                "agent": agent.name,
                "attempt": attempt,
                "raw_output": output,
                "raw_lines": raw_lines,
                "parse_error": parse_result.error,
                "parsed": parse_result.data,
                "satisfaction": parse_result.data.get("satisfaction")
                if parse_result.data
                else None,
                "key_points": parse_result.data.get("key_points")
                if parse_result.data
                else None,
                "needs_from_other": parse_result.data.get("needs_from_other")
                if parse_result.data
                else None,
            }
            if parse_result.error:
                self.transcript.append(entry)
                print("[JSON 解析失敗，將重試一次]" if attempt == 1 else "[JSON 解析失敗，已標記 error]")
                if attempt == 2:
                    self.context.append(entry)
            else:
                self.transcript.append(entry)
                self.context.append(entry)
                print(f"[{agent.name} satisfaction: {entry['satisfaction']}]")
                return entry
        print(f"[{agent.name} satisfaction: N/A]")
        return entry

    def _summarize_and_trim_context(self) -> Tuple[Optional[str], List[Dict]]:
        if not self.context:
            return None, []
        keep = max(0, self.summary_keep_last)
        if keep == 0:
            recent = []
            older = self.context
        else:
            recent = self.context[-keep:]
            older = self.context[:-keep]
        if not older:
            return None, recent
        points: List[str] = []
        for entry in older:
            if entry.get("parse_error"):
                continue
            parsed = entry.get("parsed") or {}
            key_points = parsed.get("key_points") or entry.get("key_points") or []
            if key_points:
                points.extend(key_points)
            else:
                reply = parsed.get("reply_zh_tw") or ""
                if reply:
                    points.append(reply)
        if not points:
            return None, recent
        if self.summary_max_points > 0:
            points = points[-self.summary_max_points :]
        summary = "\n".join(f"- {p}" for p in points)
        return summary, recent

    def _parse_output(self, text: str) -> ParseResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return ParseResult(data=None, error=str(exc))

        if not isinstance(data, dict):
            return ParseResult(data=None, error="JSON is not an object.")

        required = ["reply_zh_tw", "satisfaction", "key_points", "needs_from_other"]
        for key in required:
            if key not in data:
                return ParseResult(data=None, error=f"Missing key: {key}")

        if not isinstance(data["reply_zh_tw"], str):
            return ParseResult(data=None, error="reply_zh_tw must be string.")
        if not isinstance(data["needs_from_other"], str):
            return ParseResult(data=None, error="needs_from_other must be string.")
        if not isinstance(data["key_points"], list) or not all(
            isinstance(item, str) for item in data["key_points"]
        ):
            return ParseResult(data=None, error="key_points must be list[str].")

        try:
            satisfaction = int(data["satisfaction"])
        except (TypeError, ValueError):
            return ParseResult(data=None, error="satisfaction must be int 0-100.")
        if satisfaction < 0 or satisfaction > 100:
            return ParseResult(data=None, error="satisfaction out of range.")

        data["satisfaction"] = satisfaction
        return ParseResult(data=data, error=None)

    def _latest_entry(self, agent_name: str) -> Optional[Dict]:
        for entry in reversed(self.transcript):
            if entry["agent"] == agent_name:
                return entry
        return None

    def _get_satisfaction(self, entry: Optional[Dict]) -> Optional[int]:
        if not entry or entry.get("parse_error"):
            return None
        return entry.get("satisfaction")

    def _write_transcript(self) -> str:
        if self.transcript_path:
            path = self.transcript_path
        else:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = f"transcript_{ts}.jsonl"

        with open(path, "w", encoding="utf-8") as f:
            for entry in self.transcript:
                safe_entry = self._json_safe(entry, path="entry")
                f.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
        return path

    def _print_final_report(
        self,
        last_valid: Dict[str, Dict],
        stop_reason: str,
        reached_target: bool,
    ) -> None:
        print("\n=== Final Report ===")
        self._print_agent_conclusion("Agent A", last_valid.get("Agent A"))
        self._print_agent_conclusion("Agent B", last_valid.get("Agent B"))
        self._print_consensus_summary(last_valid)

        if not reached_target:
            reasons = self._infer_unmet_reasons()
            print("\n未達成原因（推論）：")
            for reason in reasons:
                print(f"- {reason}")

        print(f"\n停止原因：{stop_reason}")

    def _print_agent_conclusion(self, agent_name: str, entry: Optional[Dict]) -> None:
        print(f"\n{agent_name} 最終結論：")
        if not entry or entry.get("parse_error"):
            print("- 無法取得有效結論（JSON 解析失敗）。")
            return
        points = entry.get("key_points") or []
        if points:
            for point in points:
                print(f"- {point}")
        else:
            reply = entry.get("parsed", {}).get("reply_zh_tw", "")
            print(f"- {reply}" if reply else "- 無內容")

    def _print_consensus_summary(self, last_valid: Dict[str, Dict]) -> None:
        print("\n系統整合共識摘要：")
        summaries: List[str] = []
        for agent_name in ["Agent A", "Agent B"]:
            entry = last_valid.get(agent_name)
            if entry and not entry.get("parse_error"):
                points = entry.get("key_points") or []
                summaries.extend(points)

        if summaries:
            seen = set()
            for item in summaries:
                if item in seen:
                    continue
                seen.add(item)
                print(f"- {item}")
        else:
            print("- 尚無有效共識摘要。")

    def _infer_unmet_reasons(self) -> List[str]:
        recent = [e for e in self.transcript if not e.get("parse_error")][-10:]
        needs: List[str] = []
        points: List[str] = []
        for entry in recent:
            need = entry.get("needs_from_other")
            if need:
                needs.append(need)
            key_points = entry.get("key_points") or []
            for point in key_points:
                points.append(point)

        reasons: List[str] = []
        if needs:
            reasons.append("近期仍有未解答的需求或澄清點：" + "；".join(needs[-5:]))
        if points:
            reasons.append("最後數輪重點仍未形成一致方向：" + "；".join(points[-5:]))
        if not reasons:
            reasons.append("近期多輪對話未能同時提升雙方滿意度。")
        return reasons

    def _json_safe(self, value, path: str):
        if isinstance(value, bytes):
            print(f"[warn] bytes detected at {path}; decoding with utf-8 replacement.")
            return value.decode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {k: self._json_safe(v, path=f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_safe(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, tuple):
            return [self._json_safe(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
        return value
