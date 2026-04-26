"""
Stage 5-7: LLM Summarization Pipeline + Structured Output Parsing
- LangChain multi-chain pipeline: per-chunk → consolidation → action items
- Supports GPT-4o (OpenAI) and Gemini (Google)
- Pydantic schema enforcement for structured JSON output
"""

import os
import json
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env file if present


# ─── Pydantic Output Schema ────────────────────────────────────────────────────

class TimestampedTopic(BaseModel):
    time: str = Field(description="Timestamp in MM:SS format")
    topic: str = Field(description="Topic or subject discussed at this time")


class SummaryOutput(BaseModel):
    summary: str = Field(description="Concise 3-5 sentence summary of the full content")
    key_points: List[str] = Field(description="5-10 key insights or important points")
    action_items: List[str] = Field(description="Concrete action items or takeaways")
    timestamps: List[TimestampedTopic] = Field(description="Major topic timestamps")


# ─── Prompts ───────────────────────────────────────────────────────────────────

SEGMENT_SUMMARY_PROMPT = """You are an expert content analyst. Summarize the following transcript segment concisely.
Focus on key information, decisions, and facts. Remove any filler content.

TRANSCRIPT SEGMENT:
{text}

Provide a concise summary (2-4 sentences):"""


CONSOLIDATION_PROMPT = """You are an expert content summarizer. Given multiple segment summaries from a transcript,
create a comprehensive final analysis.

SEGMENT SUMMARIES:
{summaries}

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "summary": "3-5 sentence overall summary",
  "key_points": ["point 1", "point 2", "point 3", "...up to 10 points"],
  "action_items": ["action 1", "action 2", "..."],
  "timestamps": [
    {{"time": "0:00", "topic": "topic name"}},
    ...
  ]
}}

Do not include any text before or after the JSON. Return ONLY the JSON object."""


# ─── Main Function ─────────────────────────────────────────────────────────────

def summarize_chunks(
    chunks: List[str],
    model_name: str = "gpt-4o",
    api_key: Optional[str]= None
) -> Optional[Dict[str, Any]]:
    """
    Run the full LangChain summarization pipeline on text chunks.

    Pipeline:
    1. Summarize each chunk individually (map step)
    2. Consolidate all summaries into structured output (reduce step)
    3. Parse and validate output with Pydantic

    Args:
        chunks: List of text chunks from text_processor
        model_name: LLM model name ('gpt-4o', 'gpt-4o-mini', 'gemini-2.5-pro')
        api_key: API key for the chosen provider

    Returns:
        Dict matching SummaryOutput schema, or None on failure
    """
    if not chunks:
        return _empty_result()
    print('api_key :',api_key)
    # Set API keys in env
    # if api_key:
    #     if "gpt" in model_name or "o1" in model_name:
    #         os.getenv["OPENAI_API_KEY"] 
    #     else:
    #         os.getenv["GOOGLE_API_KEY"] = api_key

    llm = _build_llm(model_name, api_key)
    if not llm:
        print("[Summarizer] Could not initialize LLM. Check API key and model name.")
        return None

    try:
        # Step 1: Map — summarize each chunk
        segment_summaries = _map_summarize_chunks(chunks, llm)

        # Step 2: Reduce — consolidate into final structured output
        result = _reduce_and_structure(segment_summaries, llm)
        return result

    except Exception as e:
        print(f"[Summarizer] Pipeline error: {e}")
        return None


def _build_llm(model_name: str, api_key: Optional[str]):
    """Initialize the appropriate LangChain LLM."""
    try:
        if "gpt" in model_name or "o1" in model_name:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=0.3,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
            )
        elif "gemini" in model_name:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.3,
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            )
        else:
            print(f"[Summarizer] Unknown model: {model_name}")
            return None
    except ImportError as e:
        print(f"[Summarizer] Import error: {e}. Install langchain-openai or langchain-google-genai.")
        return None
    except Exception as e:
        print(f"[Summarizer] LLM init error: {e}")
        return None


def _map_summarize_chunks(chunks, llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=SEGMENT_SUMMARY_PROMPT
    )

    chain = prompt | llm
    summaries = []

    for chunk in chunks:
        res = chain.invoke({"text": chunk[:3000]})
        summaries.append(res.content if hasattr(res, "content") else str(res))

    return summaries

def _reduce_and_structure(summaries: List[str], llm) -> Optional[Dict[str, Any]]:
    """Consolidate chunk summaries into final structured output."""
    try:
        combined = "\n\n---\n\n".join(
            f"Segment {i+1}:\n{s}" for i, s in enumerate(summaries)
        )

        # Truncate if too long
        if len(combined) > 12000:
            combined = combined[:12000] + "\n\n[... truncated for context limit ...]"

        prompt = PromptTemplate(
            input_variables=["summaries"],
            template=CONSOLIDATION_PROMPT
        )

        # ✅ New chain style (no LLMChain)
        chain = prompt | llm

        response = chain.invoke({"summaries": combined})

        # Extract text safely
        raw_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        # Parse and validate JSON
        return _parse_structured_output(raw_text)

    except Exception as e:
        print(f"[Summarizer] Reduce step failed: {e}")
        return None


def _parse_structured_output(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response and validate with Pydantic."""
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()

    try:
        data = json.loads(text)
        validated = SummaryOutput(**data)
        return validated.dict()
    except json.JSONDecodeError as e:
        print(f"[Summarizer] JSON parse error: {e}")
        # Try to extract JSON from response
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                validated = SummaryOutput(**data)
                return validated.dict()
            except Exception:
                pass
        return _fallback_parse(text)
    except Exception as e:
        print(f"[Summarizer] Pydantic validation error: {e}")
        return _fallback_parse(text)


def _fallback_parse(text: str) -> Dict[str, Any]:
    """Generate a basic result dict if structured parsing fails."""
    return {
        "summary": text[:500] if text else "Summary could not be generated.",
        "key_points": ["See full transcript for details."],
        "action_items": [],
        "timestamps": [],
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "summary": "No content to summarize.",
        "key_points": [],
        "action_items": [],
        "timestamps": [],
    }


def _extract_text(response: Any) -> str:
    """Extract text string from various LangChain response formats."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("text", response.get("output", str(response)))
    if hasattr(response, "content"):
        return response.content
    return str(response)
