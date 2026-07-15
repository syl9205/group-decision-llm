"""
Single-Prompt Baseline Experiment
=================================
Response to Reviewer #1 Comment (1): 4-step sequential framework vs single-prompt comparison

Existing pipeline: Step1 -> Step2 -> Step3 -> Step4 (a separate API call each, passing previous results)
This experiment: perform Steps 1~4 at once with a single prompt + a single function call

Unified by using the CoT (Chain-of-Thought) prompt at every Step.
"""

import json
import os
import logging
import time
import copy
import nbformat
import pandas as pd
import numpy as np
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key from the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# System Prompt (same as before)
# ============================================================
COMMON_SYSTEM_PROMPT = """
You are an AI language model tasked with analyzing Japanese conversations where a group decides on a final food destination.
Your role is to uncover how individual characteristics and their interactions contribute to the group's decision-making process.
Focus on how personal traits, preferences, and existing relationships influence the flow of the conversation and lead to the final decision.

Important Considerations:
- Pay careful attention to the Japanese cultural context, including colloquialisms, abbreviations, and slang commonly found in informal conversations.
- Be mindful that participants may have shared knowledge or assumptions not explicitly mentioned in the dialogue.
- Interpret not just what is explicitly said, but also implied meanings based on shared history, tone, or familiarity between participants.
- Maintain objectivity in your analysis while being sensitive to cultural nuances.
- Your ultimate goal is to trace how individual interactions shape the group's final decision on the food destination.
- In Conversation Text Data(Input), there are CONVERSATION PART and INFORMATION PART. INFORMARION PART includes `Website Link` and `Restaurant`. `Website Link` expresses website link appeared in CONVERSATION PART. `Restaurant` expresses the name of the restaurant corresponding to `Website Link`.
- Some `Website Link` is blank because some restaurant names are expressed with no link.
"""

# ============================================================
# Unified Single Prompt (CoT for all steps)
# ============================================================
SINGLE_PROMPT_COT = """The following is a conversation in Japanese between a group of friends trying to choose a restaurant to eat-out. Please analyse this conversation **in a single pass** to complete all four analysis steps below. As you perform the analysis, explain your reasoning process briefly and concisely in Japanese at each step.

---
# Step 1.1: Initial Setup
Extract Key Information.

- `Participant`: List all participants' names.
- `Restaurant`: List all restaurant names mentioned. Extract `Restaurant` mentioned in the CONVERSATION PART and extract the "exact name" referencing the **INFORMATION PART**.
- `Final`: Identify and extract the chosen restaurant in the **INFORMATION PART**.

---
# Step 1.2: Individual Characteristics Analysis

## 1.2.1 Suggestion
Analyze suggestion's strongness. Suggestion means claims of preference.
- `Egocentrism` Types: Strong, Moderate, Weak
- `Strong`: Propose one's preference in many times. Insist on one's preference with strong expressions.
- `Moderate`: Propose one's preference, but not many times. Insist on one's preference with normal expressions.
- `Weak`: Don't propose one's preference, have no ideas.

## 1.2.2 Response
Analyze response's attribute. Response means statements in response to other people's claims of preference.
- `Egocentrism` Types: Agreeable, Moderate, Disagreeable
- `Agreeable`: **Often** agree with other's suggestion and change one's mind to follow the others.
- `Moderate`: **Not too much** tendency to agree and disagree.
- `Disagreeable`: **Often** disagree with other's suggestion and don't change one's mind easily.

---
# Step 2: Mention Analysis

Using the results from Steps 1.1 and 1.2, perform a detailed analysis to determine which participants first mentioned each restaurant.

**For each line of the conversation:**
  - Identify if a restaurant is mentioned.
  - Note which participant mentioned it.
  - Determine if this is the first time the restaurant is suggested.
  - Record any context or reasoning provided.

**Construct a chain of thought:**
  - Step-by-step, build upon the information gathered to map out who first mentioned each restaurant.
  - Include your reasoning at each step to show how you identified the initial mentions.

**Final Output:**
MentionedTable
- Rows: Participants extracted in Step 1.1.
- Columns: Restaurants extracted in Step 1.1.
- Fill each cell with "Mentioned" if the participant **first mentioned** the restaurant, or "None" if not.

---
# Step 3: Perception Analysis (Emotion-Focused)

Using the results from Steps 1 and 2, perform a detailed analysis to determine the **emotional tone** of each participant's expression towards each restaurant mentioned in the conversation.
Focus exclusively on **emotional tone** (Positive, Negative, or Neutral) and do not consider external restrictions (e.g., restaurant availability or operational status) in your analysis.

**For each participant-restaurant pair:**
Review the conversation to identify the emotional tone (**Positive**, **Negative**, or **Neutral**) of all expressions made by the participants with respect to the restaurant.

- Apply the following rules:
    1. **Consistent Emotional Tone:**
       - Always **Positive** → record **"Positive"**.
       - Always **Negative** → record **"Negative"**.
       - Only **Neutral** or no emotion → record **"Neutral"**.
    2. **Change in Emotional Tone:**
       - Both **Positive** and **Negative** → record **"Mix"**.

**Conduct a comprehensive perception analysis:**
  - Step by step, document how you determined each participant's emotional tone for each restaurant.
  - Include quotations or references to specific parts of the conversation.

**Output:**
PerceptionTable
- Rows: Participants. Columns: Restaurants.
- Cell Values: "Positive", "Negative", "Neutral", "Mix"

---
# Step 4: Perception Interpretation

Using the results from Steps 1 to 3, perform a detailed analysis to determine each participant's **preferences** and **constraints** regarding each restaurant.

**Factor Definitions:**
    **A1: Restaurant Quality** - Food quality, food genre, service quality, ambiance, seating capacity.
    **A2: Accessibility and Location** - Access convenience, attractiveness of surrounding area.
    **A3: Schedule constraints** - Group schedules, business hours, reservation slots.
    **A4: Social Utility for Consensus** - Preferences affected by other members' preferences.
    **A5: Inertia** - Prior experience, familiarity, variety-seeking behavior.
    **A6: Economic Considerations** - Price range, budget constraints, cost-effectiveness.
    **A7: Others** - Factors other than above, or no specific evidence.

**Construct a comprehensive analysis:**
- Step by step, document how you determined each factor for each participant-restaurant pair.
- Include quotations or references to specific parts of the conversation.

**Output:**
PreferenceTable & ConstraintTable
- Rows: Participants. Columns: Restaurants.
- Cell Values: Factor codes (e.g., "A1", "A2") or "None".

---
**Important Considerations:**
- Pay careful attention to the Japanese cultural context, including colloquialisms, abbreviations, and slang.
- Consider shared knowledge or assumptions not explicitly mentioned.
- Interpret implied meanings based on shared history, tone, or familiarity.
- At each step, explain your reasoning process briefly and concisely in Japanese.
- All outputs should be selected only from the provided options.
- Please adhere to the output format.
"""

# ============================================================
# Unified Function Tool (all Step outputs combined into one)
# ============================================================
SINGLE_FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "complete_analysis",
        "description": "Outputs the complete analysis results for all four steps: participant/restaurant extraction, egocentrism analysis, mention analysis, perception analysis, and preference/constraint interpretation.",
        "parameters": {
            "type": "object",
            "required": [
                "participants", "restaurant_brands", "final_restaurant",
                "suggestion_table", "response_table",
                "mentioned_table",
                "sentiment_table",
                "preference_table", "constraint_table"
            ],
            "properties": {
                # === Step 1.1 ===
                "participants": {
                    "type": "array",
                    "description": "A list of all anonymized participant names (e.g., A, B, C).",
                    "items": {"type": "string"}
                },
                "restaurant_brands": {
                    "type": "array",
                    "description": "All restaurant brand names mentioned, matched to INFORMATION PART.",
                    "items": {"type": "string"}
                },
                "final_restaurant": {
                    "type": "string",
                    "description": "The exact name of the restaurant that was ultimately chosen."
                },
                # === Step 1.2 ===
                "suggestion_table": {
                    "type": "array",
                    "description": "Egocentrism analysis of each participant's suggestion strength.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "suggestion_type", "reasoning"],
                        "properties": {
                            "participant": {"type": "string"},
                            "suggestion_type": {
                                "type": "string",
                                "enum": ["Strong", "Moderate", "Weak"]
                            },
                            "reasoning": {"type": "string"}
                        },
                        "additionalProperties": False
                    }
                },
                "response_table": {
                    "type": "array",
                    "description": "Egocentrism analysis of each participant's response attribute.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "response_type", "reasoning"],
                        "properties": {
                            "participant": {"type": "string"},
                            "response_type": {
                                "type": "string",
                                "enum": ["Agreeable", "Moderate", "Disagreeable"]
                            },
                            "reasoning": {"type": "string"}
                        },
                        "additionalProperties": False
                    }
                },
                # === Step 2 ===
                "mentioned_table": {
                    "type": "array",
                    "description": "Table showing which participant first mentioned each restaurant.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "restaurant", "mention"],
                        "properties": {
                            "participant": {"type": "string"},
                            "restaurant": {"type": "string"},
                            "mention": {
                                "type": "string",
                                "enum": ["Mentioned", "None"]
                            }
                        },
                        "additionalProperties": False
                    }
                },
                # === Step 3 ===
                "sentiment_table": {
                    "type": "array",
                    "description": "Each participant's emotional tone toward each restaurant.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "restaurant", "sentiment"],
                        "properties": {
                            "participant": {"type": "string"},
                            "restaurant": {"type": "string"},
                            "sentiment": {
                                "type": "string",
                                "enum": ["Positive", "Negative", "Neutral", "Mix"]
                            }
                        },
                        "additionalProperties": False
                    }
                },
                # === Step 4 ===
                "preference_table": {
                    "type": "array",
                    "description": "Each participant's preference factors for each restaurant.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "restaurant", "preferences"],
                        "properties": {
                            "participant": {"type": "string"},
                            "restaurant": {"type": "string"},
                            "preferences": {
                                "type": "string",
                                "description": "Comma-separated factor codes (A1-A7) or 'None'."
                            }
                        },
                        "additionalProperties": False
                    }
                },
                "constraint_table": {
                    "type": "array",
                    "description": "Each participant's constraint factors for each restaurant.",
                    "items": {
                        "type": "object",
                        "required": ["participant", "restaurant", "constraints"],
                        "properties": {
                            "participant": {"type": "string"},
                            "restaurant": {"type": "string"},
                            "constraints": {
                                "type": "string",
                                "description": "Comma-separated factor codes (A1-A7) or 'None'."
                            }
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        }
    }
}


# ============================================================
# Per-model configuration
# ============================================================
MODEL_CONFIGS = {
    "gpt5_medium": {
        "model": "gpt-5",
        "api_params": {
            "reasoning": {"effort": "medium"},
            "max_completion_tokens": 65536,  # GPT-5: generous, includes reasoning tokens
        },
        "results_dir": "appendix/results/single_prompt/gpt5/raw",
    },
    "gpt4o": {
        "model": "gpt-4o",
        "api_params": {
            "temperature": 0,
            "max_completion_tokens": 16384,
        },
        "results_dir": "appendix/results/single_prompt/gpt4o/raw",
    },
}


# ============================================================
# Utility functions
# ============================================================
def read_conversation_from_file(file_path):
    """Load the conversation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading conversation file: {e}")
        return None


def load_gold_answers(gold_dir):
    """Load the gold answers for all Steps at once."""
    gold = {}
    step_files = {
        "step1_1": "step1_1_gold.json",
        "step1_2": "step1_2_gold.json",
        "step2": "step2_gold.json",
        "step3": "step3_gold.json",
        "step4": "step4_gold.json",
    }
    for key, fname in step_files.items():
        fpath = os.path.join(gold_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                gold[key] = json.load(f)
        else:
            logger.warning(f"Gold file not found: {fpath}")
    return gold


def _ensure_tool_call(resp, expected_name: str):
    """Extract the arguments string of the first function call from a Chat Completions response."""
    try:
        msg = resp.choices[0].message
    except Exception as e:
        raise RuntimeError(f"Invalid response structure: {e}")

    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        raise RuntimeError(f"Model did not call the expected function '{expected_name}'.")

    tc = tool_calls[0]
    try:
        fname = tc.function.name
        if expected_name and fname != expected_name:
            logger.warning(f"Expected tool '{expected_name}', but got '{fname}'.")
        args_str = tc.function.arguments
    except Exception as e:
        raise RuntimeError(f"Malformed tool call: {e}")

    if not args_str or not isinstance(args_str, str):
        raise RuntimeError("Tool call has empty/invalid arguments.")

    return args_str


# ============================================================
# Core: single-prompt API call
# ============================================================
def run_single_prompt(conversation, model_config, max_retries=3):
    """
    Perform Steps 1~4 in one shot with a single prompt.

    Returns:
        dict: unified result (participants, restaurant_brands, ..., constraint_table)
    """
    model = model_config["model"]
    api_params = model_config["api_params"]

    system_prompt = f"{COMMON_SYSTEM_PROMPT}\n\nConversation:\n{conversation}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": SINGLE_PROMPT_COT}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{model}] Attempt {attempt}/{max_retries}")

            # Build API call parameters
            call_params = {
                "model": model,
                "messages": messages,
                "tools": [SINGLE_FUNCTION_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "complete_analysis"}},
                "parallel_tool_calls": False,
            }

            # Add per-model parameters
            for key, val in api_params.items():
                call_params[key] = val

            response = openai_client.chat.completions.create(**call_params)

            # Log usage
            usage = response.usage
            if usage:
                logger.info(f"  Tokens - prompt: {usage.prompt_tokens}, "
                          f"completion: {usage.completion_tokens}, "
                          f"total: {usage.total_tokens}")
                # For GPT-5, check reasoning tokens
                if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        logger.info(f"  Reasoning tokens: {details.reasoning_tokens}")

            args_str = _ensure_tool_call(response, "complete_analysis")
            result = json.loads(args_str)

            # Basic validation
            required_keys = ["participants", "restaurant_brands", "final_restaurant",
                           "suggestion_table", "response_table", "mentioned_table",
                           "sentiment_table", "preference_table", "constraint_table"]
            missing = [k for k in required_keys if k not in result]
            if missing:
                logger.warning(f"  Missing keys in result: {missing}")
                if attempt < max_retries:
                    continue

            return result, usage

        except Exception as e:
            logger.error(f"  Error on attempt {attempt}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    return None, None


# ============================================================
# Split results into the existing format (compatible with the existing evaluate functions)
# ============================================================
def split_result_to_steps(unified_result):
    """
    Convert the unified result into the existing pipeline's per-step format.
    Made compatible with the existing evaluation functions.
    """
    step1_result = {
        "participants": unified_result.get("participants", []),
        "restaurant_brands": unified_result.get("restaurant_brands", []),
        "final_restaurant": unified_result.get("final_restaurant", ""),
        "suggestion_table": unified_result.get("suggestion_table", []),
        "response_table": unified_result.get("response_table", []),
    }

    step2_result = {
        "mentioned_table": unified_result.get("mentioned_table", []),
    }

    step3_result = {
        "sentiment_table": unified_result.get("sentiment_table", []),
    }

    step4_result = {
        "preference_table": unified_result.get("preference_table", []),
        "constraint_table": unified_result.get("constraint_table", []),
    }

    return step1_result, step2_result, step3_result, step4_result


# ============================================================
# Evaluation functions (reusing existing code)
# ============================================================
def calculate_f1_list(gold_list, pred_list):
    """List-based F1 computation."""
    gold_set = set(gold_list)
    pred_set = set(pred_list)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def parse_factors(factor_str):
    """Parse a Factor string into a set."""
    if not factor_str or factor_str.strip().lower() == 'none':
        return set()
    factors = set()
    for item in factor_str.replace(' ', '').split(','):
        item = item.strip()
        if item and item.lower() != 'none':
            factors.add(item.upper())
    return factors


def evaluate_step1(gold, pred):
    """Step 1 evaluation (1.1 + 1.2)."""
    metrics = {}

    # Step 1.1
    gold_11 = gold.get("step1_1", {})
    metrics['f1_participants'] = calculate_f1_list(
        gold_11.get('participants', []), pred.get('participants', []))
    metrics['f1_restaurants'] = calculate_f1_list(
        gold_11.get('restaurant_brands', []), pred.get('restaurant_brands', []))
    metrics['final_restaurant_match'] = 1.0 if pred.get('final_restaurant', '') == gold_11.get('final_restaurant', '') else 0.0
    metrics['step11_score'] = (metrics['f1_participants'] + metrics['f1_restaurants'] + metrics['final_restaurant_match']) / 3

    # Step 1.2
    gold_12 = gold.get("step1_2", {})

    # Suggestion F1
    gold_sugg = {item['participant']: item['suggestion_type'] for item in gold_12.get('suggestion_table', [])}
    pred_sugg = {item['participant']: item['suggestion_type'] for item in pred.get('suggestion_table', [])}
    common_participants = set(gold_sugg.keys()) & set(pred_sugg.keys())
    if common_participants:
        correct = sum(1 for p in common_participants if gold_sugg[p] == pred_sugg[p])
        metrics['f1_suggestion'] = correct / max(len(gold_sugg), len(pred_sugg))
    else:
        metrics['f1_suggestion'] = 0.0

    # Response F1
    gold_resp = {item['participant']: item['response_type'] for item in gold_12.get('response_table', [])}
    pred_resp = {item['participant']: item['response_type'] for item in pred.get('response_table', [])}
    common_participants = set(gold_resp.keys()) & set(pred_resp.keys())
    if common_participants:
        correct = sum(1 for p in common_participants if gold_resp[p] == pred_resp[p])
        metrics['f1_response'] = correct / max(len(gold_resp), len(pred_resp))
    else:
        metrics['f1_response'] = 0.0

    metrics['step12_score'] = (metrics['f1_suggestion'] + metrics['f1_response']) / 2
    metrics['Step1_score'] = (metrics['step11_score'] + metrics['step12_score']) / 2

    return metrics


def evaluate_step2(gold, pred, restaurants_in_eval):
    """Step 2 evaluation (Mentioned Table F1)."""
    gold_table = gold.get("step2", {}).get("mentioned_table", [])
    pred_table = pred.get("mentioned_table", [])

    # Build lookup
    def build_lookup(table):
        lookup = {}
        for item in table:
            key = (item.get('participant', ''), item.get('restaurant', ''))
            lookup[key] = item.get('mention', 'None')
        return lookup

    gold_lk = build_lookup(gold_table)
    pred_lk = build_lookup(pred_table)

    # Calculate F1 for "Mentioned" class
    tp = fp = fn = 0
    all_keys = set(gold_lk.keys()) | set(pred_lk.keys())
    for key in all_keys:
        g = gold_lk.get(key, 'None')
        p = pred_lk.get(key, 'None')
        if g == 'Mentioned' and p == 'Mentioned':
            tp += 1
        elif g != 'Mentioned' and p == 'Mentioned':
            fp += 1
        elif g == 'Mentioned' and p != 'Mentioned':
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'F1_Mentioned': f1, 'step2_score': f1}


def evaluate_step3(gold, pred, restaurants_in_eval):
    """Step 3 evaluation (Perception F1 - macro average)."""
    gold_table = gold.get("step3", {}).get("perception_table",
                  gold.get("step3", {}).get("sentiment_table", []))
    pred_table = pred.get("sentiment_table", [])

    def build_lookup(table):
        lookup = {}
        for item in table:
            key = (item.get('participant', ''), item.get('restaurant', ''))
            # sentiment_table uses 'sentiment', perception_table might use 'perception'
            val = item.get('sentiment', item.get('perception', 'Neutral'))
            lookup[key] = val
        return lookup

    gold_lk = build_lookup(gold_table)
    pred_lk = build_lookup(pred_table)

    categories = ['Positive', 'Negative', 'Neutral', 'Mix']
    f1_per_cat = []

    all_keys = set(gold_lk.keys()) | set(pred_lk.keys())

    for cat in categories:
        tp = fp = fn = 0
        for key in all_keys:
            g = gold_lk.get(key, 'Neutral')
            p = pred_lk.get(key, 'Neutral')
            if g == cat and p == cat:
                tp += 1
            elif g != cat and p == cat:
                fp += 1
            elif g == cat and p != cat:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_cat.append(f1)

    macro_f1 = np.mean(f1_per_cat) if f1_per_cat else 0
    return {'F1_total': macro_f1, 'step3_score': macro_f1}


def evaluate_step4(gold, pred, restaurants_in_eval):
    """Step 4 evaluation (Preference/Constraint Factor F1)."""
    gold_step4 = gold.get("step4", {})

    def calc_factor_f1(gold_table, pred_table, factor_key):
        def build_lookup(table):
            lookup = {}
            for item in table:
                key = (item.get('participant', ''), item.get('restaurant', ''))
                lookup[key] = parse_factors(item.get(factor_key, 'None'))
            return lookup

        gold_lk = build_lookup(gold_table)
        pred_lk = build_lookup(pred_table)

        all_keys = set(gold_lk.keys()) | set(pred_lk.keys())
        tp = fp = fn = 0

        for key in all_keys:
            g = gold_lk.get(key, set())
            p = pred_lk.get(key, set())
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    # Jaccard similarity
    def calc_jaccard(gold_table, pred_table, factor_key):
        def build_lookup(table):
            lookup = {}
            for item in table:
                key = (item.get('participant', ''), item.get('restaurant', ''))
                lookup[key] = parse_factors(item.get(factor_key, 'None'))
            return lookup

        gold_lk = build_lookup(gold_table)
        pred_lk = build_lookup(pred_table)

        all_keys = set(gold_lk.keys()) | set(pred_lk.keys())
        scores = []
        for key in all_keys:
            g = gold_lk.get(key, set())
            p = pred_lk.get(key, set())
            if len(g | p) == 0:
                scores.append(1.0)
            else:
                scores.append(len(g & p) / len(g | p))

        return np.mean(scores) if scores else 0

    gold_pref = gold_step4.get("preference_table", [])
    gold_cons = gold_step4.get("constraint_table", [])
    pred_pref = pred.get("preference_table", [])
    pred_cons = pred.get("constraint_table", [])

    pref_f1 = calc_factor_f1(gold_pref, pred_pref, 'preferences')
    cons_f1 = calc_factor_f1(gold_cons, pred_cons, 'constraints')
    pref_js = calc_jaccard(gold_pref, pred_pref, 'preferences')
    cons_js = calc_jaccard(gold_cons, pred_cons, 'constraints')

    return {
        'preference_f1': pref_f1,
        'constraint_f1': cons_f1,
        'preference_js': pref_js,
        'constraint_js': cons_js,
        'Total_Factor_F1': (pref_f1 + cons_f1) / 2,
        'Total_JS_Score': (pref_js + cons_js) / 2,
    }


# ============================================================
# Main execution
# ============================================================
def run_experiment(model_key, conversation_files, gold_base_dir, output_base_dir, num_iterations=5):
    """
    Run the full experiment with a specific model configuration.

    Args:
        model_key: a key of MODEL_CONFIGS (e.g. "gpt5_medium")
        conversation_files: [(conv_path, gold_dir, log_name), ...]
        output_base_dir: base directory for saving results
        num_iterations: number of iterations
    """
    config = MODEL_CONFIGS[model_key]
    results_dir = os.path.join(output_base_dir, config["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"=" * 60)
    logger.info(f"Starting Single-Prompt experiment: {model_key}")
    logger.info(f"Model: {config['model']}, Params: {config['api_params']}")
    logger.info(f"Output: {results_dir}")
    logger.info(f"=" * 60)

    all_metrics = []

    for conv_path, gold_dir, log_name in conversation_files:
        logger.info(f"\n--- Processing: {log_name} ---")

        # Load conversation
        conversation = read_conversation_from_file(conv_path)
        if not conversation:
            logger.error(f"Skipping {log_name}: cannot read conversation")
            continue

        # Load gold answers
        gold = load_gold_answers(gold_dir)
        if not gold:
            logger.error(f"Skipping {log_name}: cannot load gold answers")
            continue

        # Output directory
        conv_output_dir = os.path.join(results_dir, log_name)
        os.makedirs(conv_output_dir, exist_ok=True)

        # Check checkpoint
        checkpoint_path = os.path.join(conv_output_dir, f"{log_name}_checkpoint.json")
        completed_iterations = set()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            completed_iterations = set(checkpoint.get("completed_iterations", []))
            logger.info(f"  Resuming: {len(completed_iterations)} iterations already done")

        iteration_results = []

        for iteration in range(1, num_iterations + 1):
            if iteration in completed_iterations:
                logger.info(f"  Iteration {iteration}: skipping (already done)")
                # Load existing results
                iter_path = os.path.join(conv_output_dir, f"{log_name}_iter{iteration}.json")
                if os.path.exists(iter_path):
                    with open(iter_path, 'r', encoding='utf-8') as f:
                        saved = json.load(f)
                    iteration_results.append(saved)
                continue

            logger.info(f"  Iteration {iteration}/{num_iterations}")

            try:
                result, usage = run_single_prompt(conversation, config)

                if result is None:
                    logger.error(f"  Iteration {iteration}: no result")
                    continue

                # Split by Step
                step1_r, step2_r, step3_r, step4_r = split_result_to_steps(result)

                # Evaluation
                restaurants_in_eval = result.get('restaurant_brands', [])

                m1 = evaluate_step1(gold, step1_r)
                m2 = evaluate_step2(gold, step2_r, restaurants_in_eval)
                m3 = evaluate_step3(gold, step3_r, restaurants_in_eval)
                m4 = evaluate_step4(gold, step4_r, restaurants_in_eval)

                iter_data = {
                    "iteration": iteration,
                    "log_name": log_name,
                    "model": config["model"],
                    "result": result,
                    "metrics": {
                        "step1": m1,
                        "step2": m2,
                        "step3": m3,
                        "step4": m4,
                    },
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    }
                }

                # Save per-iteration results
                iter_path = os.path.join(conv_output_dir, f"{log_name}_iter{iteration}.json")
                with open(iter_path, 'w', encoding='utf-8') as f:
                    json.dump(iter_data, f, indent=2, ensure_ascii=False)

                iteration_results.append(iter_data)

                # Update checkpoint
                completed_iterations.add(iteration)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({"completed_iterations": list(completed_iterations)}, f)

                logger.info(f"  Step1={m1['Step1_score']:.3f}, Step2={m2['step2_score']:.3f}, "
                          f"Step3={m3['step3_score']:.3f}, Step4_F1={m4['Total_Factor_F1']:.3f}")

            except Exception as e:
                logger.error(f"  Iteration {iteration} failed: {e}")
                continue

            # Rate limiting
            time.sleep(1)

        # Per-conversation summary
        if iteration_results:
            summary = compute_conversation_summary(log_name, iteration_results)
            summary_path = os.path.join(conv_output_dir, f"{log_name}_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            all_metrics.append(summary)

    # Overall experiment summary
    if all_metrics:
        save_experiment_summary(all_metrics, model_key, results_dir)

    return all_metrics


def compute_conversation_summary(log_name, iteration_results):
    """Summarize per-conversation iteration results (mean, std)."""
    summary = {"log_name": log_name, "num_iterations": len(iteration_results)}

    metric_keys = {
        "step1": ["f1_participants", "f1_restaurants", "final_restaurant_match",
                   "step11_score", "f1_suggestion", "f1_response", "step12_score", "Step1_score"],
        "step2": ["F1_Mentioned", "step2_score"],
        "step3": ["F1_total", "step3_score"],
        "step4": ["preference_f1", "constraint_f1", "preference_js", "constraint_js",
                   "Total_Factor_F1", "Total_JS_Score"],
    }

    for step, keys in metric_keys.items():
        summary[step] = {}
        for key in keys:
            values = [r["metrics"][step].get(key, 0) for r in iteration_results if step in r.get("metrics", {})]
            if values:
                summary[step][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "values": [float(v) for v in values],
                }

    # Token usage summary
    token_totals = [r.get("usage", {}).get("total_tokens", 0) for r in iteration_results]
    summary["token_usage"] = {
        "mean_total": float(np.mean(token_totals)) if token_totals else 0,
        "total_all_iterations": sum(token_totals),
    }

    return summary


def save_experiment_summary(all_metrics, model_key, results_dir):
    """Save the overall experiment summary as CSV + JSON."""
    # Save full JSON
    json_path = os.path.join(results_dir, f"experiment_summary_{model_key}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # CSV summary (per-conversation average performance)
    rows = []
    for m in all_metrics:
        row = {"log_name": m["log_name"]}
        for step in ["step1", "step2", "step3", "step4"]:
            if step in m:
                for key, val in m[step].items():
                    if isinstance(val, dict) and "mean" in val:
                        row[f"{step}_{key}_mean"] = val["mean"]
                        row[f"{step}_{key}_std"] = val["std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, f"experiment_summary_{model_key}.csv")
    df.to_csv(csv_path, index=False)

    # Print overall averages
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT SUMMARY: {model_key}")
    logger.info(f"{'='*60}")
    for col in df.columns:
        if col.endswith('_mean') and ('score' in col or 'F1' in col.lower()):
            logger.info(f"  {col}: {df[col].mean():.4f} (±{df[col].std():.4f})")


# ============================================================
# Run script
# ============================================================
def get_conversation_list(data_dir, gold_base_dir):
    """Build the list of conversation files from the data directory."""
    conversations = []
    log_dir = os.path.join(data_dir, "logs")
    answer_dir = os.path.join(data_dir, "gold")

    if not os.path.exists(log_dir):
        logger.error(f"Log directory not found: {log_dir}")
        return conversations

    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith('.txt'):
            log_name = fname.replace('.txt', '')
            conv_path = os.path.join(log_dir, fname)
            gold_dir = os.path.join(answer_dir, log_name)

            if os.path.exists(gold_dir):
                conversations.append((conv_path, gold_dir, log_name))
            else:
                logger.warning(f"No gold dir for {log_name}")

    return conversations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-Prompt Baseline Experiment")
    parser.add_argument("--model", type=str, default="gpt5_medium",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model configuration to use")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory path")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Base output directory")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per conversation")
    parser.add_argument("--conversations", type=str, nargs='*', default=None,
                       help="Specific conversation names to process (e.g., 9_log 11_log)")

    args = parser.parse_args()

    # Build conversation list
    all_conversations = get_conversation_list(args.data_dir, args.data_dir)

    # Select specific conversations only
    if args.conversations:
        all_conversations = [c for c in all_conversations if c[2] in args.conversations]

    logger.info(f"Found {len(all_conversations)} conversations to process")

    # Run experiment
    run_experiment(
        model_key=args.model,
        conversation_files=all_conversations,
        output_base_dir=args.output_dir,
        num_iterations=args.iterations,
    )
