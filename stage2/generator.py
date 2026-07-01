#!/usr/bin/env python3
"""
generator.py — Vedaz fine-tune data generator

Given a list of topics/situations, asks an LLM to produce a full chat in
Vedaz's voice and format, validates the JSON, runs it through checker.py's
safety rules, and keeps only the chats that pass.

Usage:
    export GROQ_API_KEY=...
    python generator.py --topics topics.txt --out output/generated_chats.jsonl --n-per-topic 1

    # or pass topics inline:
    python generator.py --topic "career delay, Hindi" --topic "marriage compatibility, skeptical user"

If GROQ_API_KEY is not set, the script runs in --mock mode automatically
(uses canned template chats) so the pipeline can still be exercised end to
end without network access. This is clearly logged so it's never mistaken
for real generations.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from groq import Groq

import checker  # reuses ChatReport-free helper functions from Task 1

SYSTEM_PROMPT_TEMPLATE = """You are generating ONE training example for Vedaz's AI Vedic Astrology
chat assistant fine-tune dataset. Output a single conversation as a JSON object with this exact shape:

{{
  "messages": [
    {{"role": "system", "content": "<a Vedaz system prompt describing the assistant's persona/rules, relevant to this topic>"}},
    {{"role": "user", "content": "<a realistic first user message>"}},
    {{"role": "assistant", "content": "<the assistant's reply, following ALL the rules below>"}},
    ... (optionally more user/assistant turns, alternating, 1-3 exchanges total)
  ]
}}

Vedaz's non-negotiable rules for the assistant's turns:
1. Never predict death, serious illness, or that someone's life will be "ruined".
2. Redirect serious health, legal, or major financial questions to a qualified professional.
3. Never use fear to sell remedies. Remedies (mantras, donations, pujas) are optional supportive
   practices, never guaranteed fixes, never framed as costing large sums to avoid disaster.
4. Be honest that astrology suggests tendencies and timing, not guaranteed outcomes.
5. Ask for birth date/time/place before chart-based analysis if not given.
6. Match the user's language register (Hindi, Hinglish, or English) naturally.
7. Be warm, specific, and end most replies with one natural follow-up question.

The topic/situation for this example is: "{topic}"

Make the user feel like a real, specific person (give them a distinct voice/concern), not a generic
question. Vary conversation length naturally (sometimes 1 exchange, sometimes 2-3 turns).

Reply with ONLY the JSON object. No markdown fences, no commentary, no extra text before or after.
"""

MOCK_TEMPLATES = {
    "default": {
        "messages": [
            {"role": "system", "content": "आप Vedaz के AI ज्योतिषी हैं। करुणामय, संतुलित, गैर-भाग्यवादी मार्गदर्शन देते हैं।"},
            {"role": "user", "content": "[MOCK] {topic} ke baare mein kuch bata sakte hain?"},
            {"role": "assistant", "content": (
                "[MOCK GENERATED — no GROQ_API_KEY set, this is a placeholder so the pipeline "
                "can be exercised end-to-end] Main aapki baat samajh sakta hun. Is vishay par jyotish "
                "kuch tendencies aur timing dikha sakta hai, par koi guarantee nahi de sakta. Agar yeh "
                "kisi gambhir health, legal ya bade financial faisle se juda hai, to kripya kisi "
                "professional se zaroor salah lijiye. Aapka janm vivaran (tithi, samay, sthaan) bata "
                "dijiye, taaki main behtar marg-darshan de sakun?"
            )},
        ]
    }
}


def call_llm(topic, model="llama-3.3-70b-versatile", max_retries=3):
    """Call the Groq API (Llama 3.3 70B) to generate one chat for a topic. Returns dict or None."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = SYSTEM_PROMPT_TEMPLATE.format(topic=topic)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                messages=[{"role": "system", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = raw.strip("` \n")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            data = json.loads(raw)
            if "messages" not in data or not isinstance(data["messages"], list):
                raise ValueError("missing/invalid 'messages' key")
            return data
        except Exception as e:
            print(f"  attempt {attempt}/{max_retries} failed for topic '{topic}': {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
    return None


def call_mock(topic):
    tpl = json.loads(json.dumps(MOCK_TEMPLATES["default"]))  # deep copy
    tpl["messages"][1]["content"] = tpl["messages"][1]["content"].format(topic=topic)
    return tpl


def validate_and_check(chat, dup_against):
    """Run structural + safety checks (reusing checker.py). Returns (ok, reasons)."""
    messages = chat.get("messages", [])
    shape_errors = checker.check_shape(messages)
    safety_flags = checker.run_safety_rules(messages)

    reasons = []
    if shape_errors:
        reasons += [f"shape: {e}" for e in shape_errors]
    if safety_flags:
        reasons += [f"safety[{rid}]: {snippet}" for rid, snippet in safety_flags]

    # near-duplicate check against already-accepted chats
    if dup_against:
        full_text = checker.normalize_for_hash(" ".join(m["content"] for m in messages))
        for other in dup_against:
            other_text = checker.normalize_for_hash(" ".join(m["content"] for m in other["messages"]))
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, full_text, other_text).ratio()
            if ratio >= 0.85:
                reasons.append(f"near-duplicate of an already-accepted chat (similarity={ratio:.2f})")
                break

    return (len(reasons) == 0), reasons


def main():
    ap = argparse.ArgumentParser(description="Generate Vedaz fine-tune chats")
    ap.add_argument("--topic", action="append", default=[], help="a topic string; repeatable")
    ap.add_argument("--topics", help="path to a text file, one topic per line")
    ap.add_argument("--n-per-topic", type=int, default=1, help="how many chats to attempt per topic")
    ap.add_argument("--out", default="output/generated_chats.jsonl")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--mock", action="store_true", help="force mock mode (no API calls)")
    args = ap.parse_args()

    topics = list(args.topic)
    if args.topics:
        with open(args.topics, encoding="utf-8") as f:
            topics += [line.strip() for line in f if line.strip()]
    if not topics:
        print("No topics given. Use --topic '...' (repeatable) or --topics path.txt", file=sys.stderr)
        sys.exit(1)

    use_mock = args.mock or "GROQ_API_KEY" not in os.environ
    if use_mock:
        print("NOTE: running in MOCK mode (no GROQ_API_KEY found). "
              "Output chats are placeholders, not real model generations.\n", file=sys.stderr)

    accepted = []
    rejected_log = []

    for topic in topics:
        for n in range(args.n_per_topic):
            chat = call_mock(topic) if use_mock else call_llm(topic, model=args.model)
            if chat is None:
                rejected_log.append((topic, "generation failed"))
                continue
            ok, reasons = validate_and_check(chat, dup_against=accepted)
            if ok:
                accepted.append(chat)
                print(f"[accept] topic='{topic}' (#{len(accepted)} accepted)")
            else:
                rejected_log.append((topic, "; ".join(reasons)))
                print(f"[reject] topic='{topic}': {'; '.join(reasons)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chat in accepted:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

    print(f"\nAccepted {len(accepted)} / {len(topics) * args.n_per_topic} attempted chats.")
    print(f"Saved to {out_path}")
    if rejected_log:
        print(f"\n{len(rejected_log)} rejected:")
        for topic, reason in rejected_log:
            print(f"  - '{topic}': {reason}")


if __name__ == "__main__":
    main()
