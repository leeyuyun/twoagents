from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentConfig:
    name: str
    system_prompt: str


def build_agent_configs(
    agent_a_role: Optional[str] = None,
    agent_b_role: Optional[str] = None,
) -> Dict[str, AgentConfig]:
    common_rules = (
        "你必須只輸出嚴格 JSON，不能有多餘文字或標點。\n"
        "輸出 JSON schema：\n"
        "{\n"
        '  "reply_zh_tw": "請先用「理由摘要：...」再用「結論：...」格式回答",\n'
        '  "satisfaction": 0-100 的整數,\n'
        '  "key_points": ["本輪要點1","本輪要點2"],\n'
        '  "needs_from_other": "希望對方下一輪回答/澄清什麼"\n'
        "}\n"
        "reply_zh_tw 內需要包含「理由摘要」與「結論」兩段，避免展開完整思考過程。\n"
        "安全守則：避免鼓勵自我傷害或危險指引；若觸及低潮與絕望，請以一般性建議引導求助資源。"
    )

    agent_a = AgentConfig(
        name="Agent A",
        system_prompt=(
            "你是 Agent A，偏存在主義，重視主觀經驗與自由選擇，但要避免空話，"
            "必須提出可操作的生活準則。"
            "主題由系統提示提供。"
            "回覆必須是繁體中文。"
            "每輪只輸出 JSON，不得有多餘字元。\n"
            f"{common_rules}"
        ),
    )

    agent_b = AgentConfig(
        name="Agent B",
        system_prompt=(
            "你是 Agent B，偏務實主義/系統思維，要求可驗證、可落地的方法，"
            "同時保持同理心。"
            "主題由系統提示提供。"
            "回覆必須是繁體中文。"
            "每輪只輸出 JSON，不得有多餘字元。\n"
            f"{common_rules}"
        ),
    )

    if agent_a_role:
        agent_a.system_prompt += f"\n角色補充：{agent_a_role}"
    if agent_b_role:
        agent_b.system_prompt += f"\n角色補充：{agent_b_role}"

    return {agent_a.name: agent_a, agent_b.name: agent_b}


def build_messages(
    agent: AgentConfig,
    transcript: List[Dict],
    initial_user_prompt: str,
    strict_json: bool,
    summary: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": agent.system_prompt}]
    if strict_json:
        messages.append(
            {
                "role": "system",
                "content": "重要：只能輸出嚴格 JSON，不能有任何額外字元。",
            }
        )
    if topic:
        messages.append({"role": "system", "content": f"主題：{topic}"})
    if summary:
        messages.append({"role": "system", "content": f"對話摘要：\n{summary}"})

    if not transcript:
        messages.append({"role": "user", "content": initial_user_prompt})
        return messages

    for entry in transcript:
        if entry["agent"] == agent.name:
            content = entry.get("raw_output", "")
            messages.append({"role": "assistant", "content": content})
            continue
        other_name = entry["agent"]
        content = entry.get("raw_output", "")
        if entry.get("parse_error"):
            content = f"（注意：對方上一輪輸出格式錯誤，以下是原始輸出）\n{content}"
        messages.append({"role": "user", "content": f"{other_name} 說：\n{content}"})
    return messages



