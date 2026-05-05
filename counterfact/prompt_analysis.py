"""
Prompt analysis engine for thinking model pipelines.

Thinking models receive long system prompts (often 15+ pages) that
prescribe how to design tool call sequences. This module analyzes
those prompts to:

  1. Parse into semantic sections — so we can attribute per-section
  2. Evaluate plan quality — does the generated plan follow the prompt?
  3. Compute section attribution — Shapley values over prompt sections
  4. Detect dead sections — instructions the model consistently ignores
  5. Detect conflicting sections — instructions that contradict each other

The key insight: instead of attributing quality to agents, we attribute
quality to sections of the system prompt. This tells teams "Section 7
contributes 40% of your output quality, but Section 12 is hurting you."

Dependencies: types only (LLM function is injected via llm_fn)
"""

import json
import re
from typing import Callable, Optional, Any

from counterfact.types import (
    PromptSection,
    PlanStep,
    PromptAnalysisResult,
    EvalResult,
)


# ═════════════════════════════════════════════════════════════════════════
# PROMPT PARSING
# Segment a long system prompt into meaningful sections.
# ═════════════════════════════════════════════════════════════════════════


def parse_prompt_sections(
    prompt: str,
    llm_fn: Callable[[str, float], str],
) -> list[PromptSection]:
    """
    Parse a system prompt into semantic sections.

    Uses an LLM to identify logical boundaries in the prompt (headings,
    numbered steps, topic shifts) and categorize each section.

    # PROMPT TUNING NOTE: This is the first step in prompt analysis.
    # Key levers:
    #   - Granularity: too many small sections = noisy attribution
    #   - Category assignment: affects which sections are tested
    #   - Section boundaries: must be clean for ablation to work
    """
    if not prompt or not prompt.strip():
        return []

    # PROMPT TUNING: Controls how the LLM segments the prompt.
    # For very long prompts (>10k chars), we may need to chunk and merge.
    analysis_prompt = f"""You are analyzing a system prompt to identify its logical sections.

SYSTEM PROMPT TO ANALYZE:
{prompt[:12000]}

Identify each distinct logical section of this prompt. For each section, provide:
- title: short descriptive name (3-6 words)
- content: the exact text of that section (preserve original wording)
- category: one of "instruction", "constraint", "methodology", "format", "context"

Categories:
- "instruction": direct behavioral directives ("Always cite sources", "Use formal tone")
- "constraint": limitations and guardrails ("Never give medical advice", "Stay under 500 words")
- "methodology": prescribed workflows and step sequences ("Step 1: analyze...", "First, check...")
- "format": output structure rules ("Return JSON", "Use bullet points")
- "context": background/role info ("You are a financial analyst", "The user is a...")

Respond with ONLY a JSON array:
[{{"title": "...", "content": "...", "category": "..."}}]"""

    try:
        response = llm_fn(analysis_prompt, 0.1)
        sections_data = _parse_json_response(response)

        if not isinstance(sections_data, list):
            sections_data = [sections_data]

        # Convert to PromptSection objects with character positions
        sections = []
        for i, s in enumerate(sections_data):
            content = s.get("content", "")
            # Find the character position in the original prompt
            char_start = prompt.find(content[:50]) if content else 0
            char_start = max(0, char_start)

            sections.append(PromptSection(
                index=i,
                title=s.get("title", f"Section {i+1}"),
                content=content,
                category=s.get("category", "instruction"),
                char_start=char_start,
                char_end=char_start + len(content),
            ))

        return sections

    except (json.JSONDecodeError, ValueError, RuntimeError, Exception):
        # Fallback: split by double newlines or numbered sections
        return _fallback_parse(prompt)


def parse_system_prompt(prompt: str) -> list[PromptSection]:
    """
    Rule-based parser for system prompts.
    
    Splits the prompt by:
      1. Markdown-style headings (# Section, ## Subsection)
      2. Numbered sections (1., 2., 3.)
      3. Double newlines (paragraph breaks)
      
    This provides a deterministic, fast way to segment prompts without LLM calls,
    especially useful for non-thinking models that have simpler, structured prompts.
    """
    sections = []

    # Try splitting by markdown headings first
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(prompt))

    if len(matches) >= 2:
        # Split by headings
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
            content = prompt[start:end].strip()
            # Try to infer category from heading
            title_lower = match.group(2).lower()
            category = "instruction"
            if "example" in title_lower or "few-shot" in title_lower or "few shot" in title_lower:
                category = "context"
            elif "constraint" in title_lower or "rule" in title_lower or "never" in title_lower:
                category = "constraint"
                
            sections.append(PromptSection(
                index=i,
                title=match.group(2).strip()[:60],
                content=content,
                category=category,
                char_start=start,
                char_end=end,
            ))
    else:
        # Split by double newlines
        paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            char_start = prompt.find(para[:50])
            
            # Simple heuristic for few-shot examples
            category = "instruction"
            if "Example:" in para or "Input:" in para and "Output:" in para:
                category = "context"
                
            sections.append(PromptSection(
                index=i,
                title=f"Section {i+1}",
                content=para,
                category=category,
                char_start=max(0, char_start),
                char_end=max(0, char_start) + len(para),
            ))

    return sections

def _fallback_parse(prompt: str) -> list[PromptSection]:
    return parse_system_prompt(prompt)


# ═════════════════════════════════════════════════════════════════════════
# PLAN QUALITY EVALUATION
# Check if the model's execution plan follows the system prompt.
# ═════════════════════════════════════════════════════════════════════════


def check_plan_quality(
    plan_steps: list[PlanStep],
    prompt_sections: list[PromptSection],
    llm_fn: Callable[[str, float], str],
) -> EvalResult:
    """
    Evaluate whether a thinking model's plan follows the system prompt.

    Checks three dimensions:
      - Completeness: does the plan cover all methodology sections?
      - Efficiency: are there redundant or unnecessary steps?
      - Adherence: does the plan follow the prescribed order?

    # PROMPT TUNING NOTE: This prompt evaluates plan-to-prompt alignment.
    # Key levers:
    #   - How strictly to enforce step ordering
    #   - Whether to penalize extra (creative) steps
    #   - How to weight methodology vs instruction compliance
    """
    if not plan_steps:
        return EvalResult(
            check_name="plan_quality",
            passed=False,
            severity="critical",
            message="No plan steps provided — model did not generate an execution plan.",
            details={"num_steps": 0},
        )

    # Format the plan and methodology sections for the LLM
    plan_text = "\n".join(
        f"Step {s.step_index}: [{s.tool_name}] {s.description}"
        for s in plan_steps
    )

    methodology = "\n".join(
        f"[{s.category}] {s.title}: {s.content[:300]}"
        for s in prompt_sections
        if s.category in ("methodology", "instruction")
    )

    # PROMPT TUNING: Controls how strictly plan adherence is evaluated.
    prompt = f"""You are evaluating whether a model's execution plan follows its system prompt instructions.

SYSTEM PROMPT INSTRUCTIONS:
{methodology[:4000]}

MODEL'S EXECUTION PLAN:
{plan_text[:3000]}

Evaluate three dimensions:
1. COMPLETENESS (0-1): Does the plan cover all required steps from the instructions?
2. EFFICIENCY (0-1): Are there unnecessary or redundant steps?
3. ADHERENCE (0-1): Does the plan follow the prescribed methodology/order?

Also identify:
- missing_steps: steps required by the prompt but absent from the plan
- extra_steps: steps in the plan not prescribed by the prompt
- order_violations: steps that are in the wrong sequence

Respond with ONLY valid JSON:
{{"completeness": 0.0-1.0, "efficiency": 0.0-1.0, "adherence": 0.0-1.0,
  "missing_steps": ["..."], "extra_steps": ["..."], "order_violations": ["..."],
  "reasoning": "brief explanation"}}"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed_res = _parse_json_response(response)
        parsed = parsed_res if isinstance(parsed_res, dict) else (parsed_res[0] if isinstance(parsed_res, list) and parsed_res else {})
        
        completeness = float(parsed.get("completeness", 0.5))
        efficiency = float(parsed.get("efficiency", 0.5))
        adherence = float(parsed.get("adherence", 0.5))

        # Weighted average: completeness matters most
        score = 0.5 * completeness + 0.2 * efficiency + 0.3 * adherence
        score = max(0.0, min(1.0, score))

        return EvalResult(
            check_name="plan_quality",
            passed=score >= 0.7,
            severity="critical" if score < 0.5 else "warning" if score < 0.7 else "info",
            message=f"Plan quality: {score:.2f} (completeness={completeness:.2f}, efficiency={efficiency:.2f}, adherence={adherence:.2f})",
            details={
                "score": round(score, 3),
                "completeness": round(completeness, 3),
                "efficiency": round(efficiency, 3),
                "adherence": round(adherence, 3),
                "missing_steps": parsed.get("missing_steps", []),
                "extra_steps": parsed.get("extra_steps", []),
                "order_violations": parsed.get("order_violations", []),
                "reasoning": parsed.get("reasoning", ""),
            },
        )

    except (json.JSONDecodeError, ValueError, RuntimeError, Exception) as e:
        return EvalResult(
            check_name="plan_quality",
            passed=False,
            severity="warning",
            message=f"Could not evaluate plan quality: {str(e)[:200]}",
            details={"error": str(e)[:200]},
        )


# ═════════════════════════════════════════════════════════════════════════
# PROMPT-SECTION ATTRIBUTION
# Compute Shapley values over prompt sections instead of agents.
# ═════════════════════════════════════════════════════════════════════════


def run_prompt_section_attribution(
    prompt: str,
    sections: list[PromptSection],
    query: str,
    original_output: str,
    llm_fn: Callable[[str, float], str],
    num_ablations: int = 1,
) -> dict[int, float]:
    """
    Compute attribution scores for each section of the system prompt.

    For each section, we ablate it (remove it from the prompt) and
    ask the LLM to simulate what the model would produce without that
    instruction. Then we compare the quality to the original output.

    Returns: {section_index: attribution_score}
      - Positive = removing this section HURTS quality (section is helpful)
      - Negative = removing this section IMPROVES quality (section is harmful)
      - Near zero = section has no measurable impact (possibly dead)

    # PROMPT TUNING NOTE: The simulation prompt controls how realistically
    # the LLM simulates section ablation. If results are too similar,
    # make the prompt more explicit about expected changes.
    """
    if not sections:
        return {}

    attributions = {}

    for section in sections:
        # Build the prompt with this section removed
        ablated_prompt = _ablate_section(prompt, section)

        # PROMPT TUNING: Controls how the LLM simulates ablated behavior.
        sim_prompt = f"""You are simulating how a model would respond if part of its system prompt was REMOVED.

ORIGINAL SYSTEM PROMPT (with section "{section.title}" REMOVED):
{ablated_prompt[:6000]}

QUERY: {query[:500]}

The model's ORIGINAL output (with all instructions) was:
{original_output[:1000]}

Now generate what the model would output WITHOUT the "{section.title}" section.
Consider: what specific behavior would change if "{section.content[:200]}" was not in the prompt?

Generate the modified output (under 200 words). Make it noticeably different if this section was important."""

        total_quality_delta = 0.0
        for _ in range(num_ablations):
            try:
                ablated_output = llm_fn(sim_prompt, 0.3)

                # Score how different the ablated output is from the original
                quality_prompt = f"""Compare these two outputs and score how much quality was LOST by removing a prompt section.

ORIGINAL OUTPUT:
{original_output[:800]}

ABLATED OUTPUT (with section "{section.title}" removed):
{ablated_output[:800]}

Score the quality difference:
- 1.0: The ablated output is MUCH worse (section was critical)
- 0.5: Some quality loss (section was somewhat important)
- 0.0: No quality difference (section had no impact)
- -0.5: The ablated output is actually BETTER (section was harmful)

Respond with ONLY valid JSON: {{"score": -1.0 to 1.0, "reasoning": "brief explanation"}}"""

                quality_response = llm_fn(quality_prompt, 0.1)
                parsed_res = _parse_json_response(quality_response)
                parsed = parsed_res if isinstance(parsed_res, dict) else (parsed_res[0] if isinstance(parsed_res, list) and parsed_res else {})
                delta = float(parsed.get("score", 0.0))
                total_quality_delta += max(-1.0, min(1.0, delta))

            except (json.JSONDecodeError, ValueError, RuntimeError, Exception):
                total_quality_delta += 0.0

        attributions[section.index] = round(total_quality_delta / max(1, num_ablations), 4)

    # Normalize absolute values to sum to 1
    total = sum(abs(v) for v in attributions.values())
    if total > 0:
        attributions = {k: round(v / total, 4) for k, v in attributions.items()}

    return attributions


def _ablate_section(prompt: str, section: PromptSection) -> str:
    """
    Remove a section from the prompt text.

    Uses character positions if available, otherwise removes by content match.
    """
    if section.char_start > 0 or section.char_end > 0:
        before = prompt[:section.char_start]
        after = prompt[section.char_end:]
        return (before + after).strip()

    # Fallback: remove by content match
    return prompt.replace(section.content, "").strip()


# ═════════════════════════════════════════════════════════════════════════
# DEAD SECTION DETECTION
# Find prompt sections the model consistently ignores.
# ═════════════════════════════════════════════════════════════════════════


def detect_dead_sections(
    sections: list[PromptSection],
    outputs: list[str],
    llm_fn: Callable[[str, float], str],
) -> list[int]:
    """
    Detect prompt sections the model consistently ignores.

    For each section, we check whether ANY of the provided outputs show
    evidence that the model followed that instruction. A section is "dead"
    if it has no observable effect across all outputs.

    This is valuable because teams often accumulate prompt instructions
    over time without removing obsolete ones — dead sections add token
    cost without any benefit.

    # PROMPT TUNING NOTE: The key challenge is distinguishing "intentionally
    # not triggered" (e.g., a constraint that wasn't relevant) from "ignored"
    # (e.g., an instruction the model never follows).
    """
    if not sections or not outputs:
        return []

    dead = []
    output_sample = "\n---\n".join(o[:500] for o in outputs[:5])

    for section in sections:
        # PROMPT TUNING: Controls sensitivity of dead section detection.
        prompt = f"""Check whether a model's outputs show evidence of following this instruction.

INSTRUCTION:
"{section.title}": {section.content[:500]}

MODEL OUTPUTS (sample):
{output_sample[:3000]}

Is there evidence that the model followed this instruction in ANY of the outputs?
- "followed": clear evidence the model applied this instruction
- "possibly_followed": ambiguous — might have been followed implicitly
- "not_followed": no evidence the model applied this instruction at all

Respond with ONLY valid JSON: {{"verdict": "followed|possibly_followed|not_followed", "reasoning": "brief explanation"}}"""

        try:
            response = llm_fn(prompt, 0.1)
            parsed_res = _parse_json_response(response)
            parsed = parsed_res if isinstance(parsed_res, dict) else (parsed_res[0] if isinstance(parsed_res, list) and parsed_res else {})
            verdict = parsed.get("verdict", "possibly_followed")

            if verdict == "not_followed":
                dead.append(section.index)

        except (json.JSONDecodeError, ValueError, RuntimeError, Exception):
            pass  # Don't flag as dead if we can't determine

    return dead


# ═════════════════════════════════════════════════════════════════════════
# CONFLICTING SECTION DETECTION
# Find prompt sections that contradict each other.
# ═════════════════════════════════════════════════════════════════════════


def detect_conflicting_sections(
    sections: list[PromptSection],
    llm_fn: Callable[[str, float], str],
) -> list[tuple[int, int]]:
    """
    Detect pairs of prompt sections that contradict each other.

    Long prompts often accumulate contradictory instructions over time
    (e.g., "Always be concise" vs "Provide detailed explanations").
    These conflicts confuse the model and reduce output quality.

    Only checks pairs where both sections are instructions or constraints
    (context sections rarely conflict with each other).

    # PROMPT TUNING NOTE: The key challenge is distinguishing true
    # contradictions from complementary instructions that apply in
    # different contexts.
    """
    if len(sections) < 2:
        return []

    conflicts = []

    # Only check instruction/constraint sections (context rarely conflicts)
    checkable = [s for s in sections if s.category in ("instruction", "constraint", "methodology")]

    # Check pairs (avoid O(n²) explosion for large prompts)
    pairs_to_check = []
    for i in range(len(checkable)):
        for j in range(i + 1, min(i + 5, len(checkable))):  # Check nearby pairs
            pairs_to_check.append((checkable[i], checkable[j]))

    if not pairs_to_check:
        return []

    # Batch check in groups to reduce LLM calls
    batch_text = "\n\n".join(
        f"PAIR {k+1}:\n  Section A ({a.title}): {a.content[:200]}\n  Section B ({b.title}): {b.content[:200]}"
        for k, (a, b) in enumerate(pairs_to_check[:10])
    )

    # PROMPT TUNING: Controls conflict detection sensitivity.
    prompt = f"""Check each pair of instructions for contradictions.

{batch_text}

For each pair, determine:
- "no_conflict": instructions are compatible or complementary
- "tension": instructions pull in different directions but aren't contradictory
- "conflict": instructions directly contradict each other

Respond with ONLY a JSON array:
[{{"pair": 1, "verdict": "no_conflict|tension|conflict", "reasoning": "..."}}]"""

    try:
        response = llm_fn(prompt, 0.1)
        parsed = _parse_json_response(response)

        if not isinstance(parsed, list):
            parsed = [parsed]

        for item in parsed:
            pair_idx = int(item.get("pair", 0)) - 1
            verdict = item.get("verdict", "no_conflict")
            if verdict == "conflict" and 0 <= pair_idx < len(pairs_to_check):
                a, b = pairs_to_check[pair_idx]
                conflicts.append((a.index, b.index))

    except (json.JSONDecodeError, ValueError, RuntimeError, Exception):
        pass

    return conflicts


# ═════════════════════════════════════════════════════════════════════════
# FULL PROMPT ANALYSIS ORCHESTRATOR
# Runs all analyses and returns a complete PromptAnalysisResult.
# ═════════════════════════════════════════════════════════════════════════


def analyze_prompt(
    prompt: str,
    query: str,
    output: str,
    llm_fn: Callable[[str, float], str],
    additional_outputs: Optional[list[str]] = None,
    plan_steps: Optional[list[PlanStep]] = None,
) -> PromptAnalysisResult:
    """
    Run complete prompt analysis: parse, attribute, detect dead/conflicts.

    This is the main entry point for thinking model evaluation. It:
      1. Parses the prompt into sections
      2. Evaluates plan quality (if plan_steps provided)
      3. Computes section attribution
      4. Detects dead sections
      5. Detects conflicting sections

    Args:
        prompt: The system prompt to analyze
        query: A sample query the model processed
        output: The model's output for that query
        llm_fn: LLM function (prompt, temperature) -> str
        additional_outputs: More outputs for dead section detection
        plan_steps: The model's execution plan (if available)
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be non-empty")
    if llm_fn is None:
        raise ValueError("llm_fn is required for prompt analysis")

    # Step 1: Parse prompt into sections
    sections = parse_prompt_sections(prompt, llm_fn)

    # Step 2: Plan quality (if plan provided)
    plan_score = 0.0
    plan_details = {}
    if plan_steps:
        plan_result = check_plan_quality(plan_steps, sections, llm_fn)
        plan_score = plan_result.details.get("score", 0.0)
        plan_details = plan_result.details

    # Step 3: Section attribution
    attributions = run_prompt_section_attribution(
        prompt, sections, query, output, llm_fn,
    )

    # Update section importance from attribution
    for section in sections:
        if section.index in attributions:
            section.importance = abs(attributions[section.index])

    # Step 4: Dead section detection
    all_outputs = [output] + (additional_outputs or [])
    dead = detect_dead_sections(sections, all_outputs, llm_fn)

    # Step 5: Conflict detection
    conflicts = detect_conflicting_sections(sections, llm_fn)

    # Compute confidence based on available data
    confidence = _compute_analysis_confidence(
        len(sections), len(all_outputs), plan_steps is not None,
    )

    return PromptAnalysisResult(
        sections=sections,
        section_attributions=attributions,
        dead_sections=dead,
        conflicting_pairs=conflicts,
        plan_quality_score=plan_score,
        plan_quality_details=plan_details,
        confidence=confidence,
    )


def score_few_shot_quality(sections: list[PromptSection]) -> dict[str, Any]:
    """
    Rule-based quality check for few-shot examples within a prompt.
    Returns a score and details about the examples found.
    """
    # Look for sections identified as examples
    example_sections = [
        s for s in sections 
        if "example" in s.title.lower() or 
           "few-shot" in s.title.lower() or
           (s.category == "context" and ("Input:" in s.content and "Output:" in s.content))
    ]
    
    if not example_sections:
        return {
            "score": 0.0,
            "has_examples": False,
            "count": 0,
            "feedback": "No few-shot examples found. Adding 3-5 high-quality examples usually improves performance."
        }
        
    total_examples = sum(s.content.lower().count("output:") for s in example_sections)
    total_examples = max(len(example_sections), total_examples)
    
    # Assess quality
    if total_examples < 3:
        score = 0.5
        feedback = f"Found {total_examples} example(s). Increasing to 3-5 distinct examples is recommended."
    elif total_examples > 10:
        score = 0.7
        feedback = f"Found {total_examples} examples. This might be excessive and consume unnecessary context. Consider curating to top 5."
    else:
        score = 1.0
        feedback = f"Found a healthy number of examples ({total_examples})."
        
    return {
        "score": score,
        "has_examples": True,
        "count": total_examples,
        "feedback": feedback
    }


def _compute_analysis_confidence(
    num_sections: int,
    num_outputs: int,
    has_plan: bool,
) -> float:
    """Estimate confidence based on available data."""
    confidence = 0.3

    # More sections = clearer structure = higher confidence
    if num_sections >= 5:
        confidence += 0.15
    elif num_sections >= 2:
        confidence += 0.1

    # More outputs = better dead section detection
    if num_outputs >= 5:
        confidence += 0.2
    elif num_outputs >= 2:
        confidence += 0.1

    # Having the plan enables plan quality analysis
    if has_plan:
        confidence += 0.15

    return min(0.95, round(confidence, 3))


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════


def _parse_json_response(text: str) -> dict | list:
    """Parse a JSON response from an LLM, handling markdown wrappers."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
