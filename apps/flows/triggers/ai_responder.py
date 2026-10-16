"""AI Knowledge-Base Responder hook.

Answers inbound messages automatically using a customizable Knowledge Base
(knowledge_base.txt) and LLM (OpenAI / Gemini).
"""

import os
import logging
from typing import Any
import httpx
from django.conf import settings

from apps.channels.events import EventType, OutboundMessage, TextBlock
from apps.flows.triggers.context import RoutingContext
from apps.flows.triggers.hooks import Consumed, HookOutcome, Passed, Stage, register_hook
from apps.messaging.models import MessageSource
from apps.messaging.services import send_outbound

logger = logging.getLogger(__name__)

KNOWLEDGE_FILE = os.path.join(settings.BASE_DIR, "knowledge_base.txt")


def load_knowledge_base() -> str:
    """Reads user-supplied training and instructions from knowledge_base.txt."""
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error("Error reading knowledge_base.txt: %s", e)
    return "Şirkətimiz müştərilərə peşəkar xidmət göstərir."


def get_ai_response(user_query: str) -> str:
    """Sends the user's message and knowledge base to OpenAI or Gemini."""
    knowledge = load_knowledge_base()
    
    system_instruction = (
        "Sən şirkətin rəsmi, ağıllı və nəzakətli virtual süni intellekt köməkçisisən.\n"
        "Müştəriyə müraciət edərkən hər zaman nəzakətli dildən ('Siz') istifadə et.\n"
        "Müştəri hansı dildə (Azərbaycan, İngilis, Rus və s.) yazırsa, həmin dildə cavab ver.\n\n"
        "Aşağıdakı BİLİK BAZASI və QAYDALARA əsasən dəqiq və aydın cavab hazırla:\n"
        "====================\n"
        f"{knowledge}\n"
        "====================\n\n"
        "Əgər müştərinin soruşduğu məlumat bilik bazasında qeyd olunmayıbsa, yalan məlumat uydurma. "
        "Nəzakətlə bildir ki, məlumatı dəqiqləşdirmək üçün operator tezliklə onunla əlaqə saxlayacaq."
    )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    # Try OpenAI
    if openai_key:
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.4,
                },
                timeout=15.0,
            )
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.exception("OpenAI call failed: %s", exc)

    # Try Google Gemini
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_instruction}\n\nMüştəri sorğusu: {user_query}"}]}
                ]
            }
            resp = httpx.post(url, json=payload, timeout=15.0)
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as exc:
            logger.exception("Gemini call failed: %s", exc)

    return (
        "Salam! Müraciətiniz üçün təşəkkür edirik. "
        "Süni intellekt köməkçimiz hazırda aktivdir və tezliklə canlı operatorumuz da sizə cavab verəcək."
    )


def ai_knowledge_responder(context: RoutingContext) -> HookOutcome:
    """DEFAULT_REPLY hook that generates an AI response based on the knowledge base."""
    if context.event.type != EventType.MESSAGE or context.contact is None:
        return Passed("not a message event")

    user_text = getattr(context.event.payload, "text", "") or ""
    if not user_text.strip():
        return Passed("empty message text")

    logger.info("AI Knowledge Responder processing message from %s: %s", context.contact.pk, user_text[:50])
    
    # Generate the reply based on the knowledge base
    reply_text = get_ai_response(user_text)

    # Send outbound message to the contact
    outbound = OutboundMessage(blocks=(TextBlock(text=reply_text),))
    event_id = getattr(context.event, "id", "") or "default"
    
    send_outbound(
        workspace=context.workspace,
        contact=context.contact,
        connection=context.connection,
        outbound=outbound,
        source=MessageSource.AUTOMATION,
        idempotency_key=f"ai_reply:{context.connection.pk}:{event_id}",
    )

    return Consumed("ai_knowledge_reply")


def register_ai_responder() -> None:
    """Register the AI knowledge responder hook."""
    register_hook(
        ai_knowledge_responder,
        stage=Stage.DEFAULT_REPLY,
        name="ai_knowledge_responder",
        priority=5,  # Runs before default 0 if configured
        replace_existing=True,
    )
