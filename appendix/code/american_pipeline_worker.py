#!/usr/bin/env python3
"""
American Pipeline - Standalone worker.
Usage: python appendix/code/american_pipeline_worker.py <conv> <prompt> <outdir> <golddir> <niter> <model> <temp>
"""

# ---- Config globals (set by init()) ----
MODEL_NAME = None
TEMPERATURE = None

# ============ Core Functions (Cell 3) ============
import json
import os
import argparse
import logging
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import copy
import nbformat  # for notebook parsing

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API configuration (key from the OPENAI_API_KEY environment variable)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai_client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = None  # Set by init()

# Define the system prompt
COMMON_SYSTEM_PROMPT = """
You are an AI language model tasked with analyzing conversations where a group decides on a final food destination.
Your role is to uncover how individual characteristics and their interactions contribute to the group's decision-making process.
Focus on how personal traits, preferences, and existing relationships influence the flow of the conversation and lead to the final decision.

Important Considerations:
- Pay careful attention to the American cultural context, including colloquialisms, abbreviations, and slang commonly found in informal conversations.
- Be mindful that participants may have shared knowledge or assumptions not explicitly mentioned in the dialogue.
- Interpret not just what is explicitly said, but also implied meanings based on shared history, tone, or familiarity between participants.
- Maintain objectivity in your analysis while being sensitive to cultural nuances.
- Your ultimate goal is to trace how individual interactions shape the group's final decision on the food destination.
- In Conversation Text Data(Input), there are CONVERSATION PART and INFORMATION PART. INFORMARION PART includes `Website Link` and `Restaurant`. `Website Link` expresses website link appeared in CONVERSATION PART. `Restaurant` expresses the name of the restaurant corresponding to `Website Link`.
- Some `Website Link` is blank because some restaurant names are expressed with no link.
"""

# FUNCTION_TOOLS dictionary
FUNCTION_TOOLS = {
    "Steps1.1-1.2": {
        "type": "function",
        "function": {
            "name": "analyze_step1",
            "description": "Extracts restaurant info, analyzes egocentrism.",
            "parameters": {
                "type": "object",
                "required": ["participants", "restaurant_brands", "final_restaurant", "suggestion_table", "response_table"],
                "properties": {
                    "participants": {
                        "type": "array",
                        "description": "A list of all anonymized participant names (e.g., A, B, C) mentioned in the conversation.",
                        "items": {
                            "type": "string",
                            "description": "An anonymized name representing a participant in the CONVERSATION PART."
                        }
                    },
                    "restaurant_brands": {
                        "type": "array",
                        "description": "A list of all restaurant brand names explicitly mentioned in the CONVERSATION PART, matched to their official names in the INFORMATION PART.",
                        "items": {
                            "type": "string",
                            "description": "Exact name of a restaurant brand as mentioned in the CONVERSATION PART, aligned with its official name from the INFORMATION PART."
                        }
                    },
                    "final_restaurant": {
                        "type": "string",
                        "description": "The exact name of the restaurant that was ultimately chosen."
                    },
                    "suggestion_table": {
                        "type": "array",
                        "description": "List analyzing how strongly each participant suggested their restaurant preference.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "suggestion_type", "reasoning"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Name of the participant"
                                },
                                "suggestion_type": {
                                    "type": "string",
                                    "description": "`Egocentrism` Type of suggestion's strongness",
                                    "enum": ["Strong", "Moderate", "Weak"]
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Justification for assigning this suggestion strength."
                                }
                            }
                        }
                    },
                    "response_table": {
                        "type": "array",
                        "description": "List analyzing how each participant responded to others' suggestions.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "response_type", "reasoning"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Name of the participant"
                                },
                                "response_type": {
                                    "type": "string",
                                    "description": "`Egocentrism` Type of response's attribute",
                                    "enum": ["Agreeable", "Moderate", "Disagreeable"]
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Justification for assigning this response type."
                                }
                            }
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    
    "Step2": {
        "type": "function",
        "function": {
            "name": "output_mentioned_table",
            "description": "Generates a MentionedTable showing which participant first mentioned each restaurant.",
            "parameters": {
                "type": "object",
                "required": ["mentioned_table"],
                "properties": {
                    "mentioned_table": {
                        "type": "array",
                        "description": "Table indicating whether each participant was the first to mention a given restaurant.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "restaurant", "mention"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Exact Name of the participant."
                                },
                                "restaurant": {
                                    "type": "string",
                                    "description": "Exact Name of the restaurant."
                                },
                                "mention": {
                                    "type": "string",
                                    "description": "Indicates whether the participant was the first to mention this restaurant.",
                                    "enum": ["Mentioned", "None"]
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    
    "Step3": {
        "type": "function",
        "function": {
            "name": "sentiment_analysis",
            "description": "Analyzes the emotional sentiment of each participant toward each restaurant, focusing solely on emotional tone.",
            "parameters": {
                "type": "object",
                "required": ["sentiment_table"],
                "properties": {
                    "sentiment_table": {
                        "type": "array",
                        "description": "Table showing each participant's emotional sentiment toward each mentioned restaurant.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "restaurant", "sentiment"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Exact Name of the participant."
                                },
                                "restaurant": {
                                    "type": "string",
                                    "description": "Exact Name of the restaurant."
                                },
                                "sentiment": {
                                    "type": "string",
                                    "description": "Final emotional sentiment of the participant toward the restaurant.",
                                    "enum": ["Positive", "Negative", "Neutral", "Mix"]
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    
    "Step4": {
        "type": "function",
        "function": {
            "name": "preference_constraint_analysis",
            "description": "Analyzes each participant's stated preferences and constraints for each restaurant based on the conversation, using predefined factor codes.",
            "parameters": {
                "type": "object",
                "required": ["preference_table", "constraint_table"],
                "properties": {
                    "preference_table": {
                        "type": "array",
                        "description": "Table showing each participant's preference factors toward each restaurant, based on their conversation statements.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "restaurant", "preferences"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Exact Name of the participant"
                                },
                                "restaurant": {
                                    "type": "string",
                                    "description": "Exact Name of the restaurant"
                                },
                                "preferences": {
                                    "type": "string",
                                    "description": "Comma-separated factor codes (e.g., A1, A2) indicating expressed preferences. Use 'None' if no preference was expressed."
                                }
                            },
                            "additionalProperties": False
                        }
                    },
                    "constraint_table": {
                        "type": "array",
                        "description": "Table showing each participant's constraint factors toward each restaurant, based on their conversation statements.",
                        "items": {
                            "type": "object",
                            "required": ["participant", "restaurant", "constraints"],
                            "properties": {
                                "participant": {
                                    "type": "string",
                                    "description": "Exact Name of the participant"
                                },
                                "restaurant": {
                                    "type": "string",
                                    "description": "Exact Name of the restaurant"
                                },
                                "constraints": {
                                    "type": "string",
                                    "description": "Comma-separated factor codes (e.g., A1, A2) indicating expressed constraints. Use 'None' if no constraint was expressed."
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
}

# Global variables

# Temperature / Model Config Helper
def get_api_kwargs():
    """Return extra kwargs for API calls based on model config"""
    kwargs = {}
    if TEMPERATURE is not None:
        kwargs['temperature'] = TEMPERATURE
    return kwargs

best_results = {}  # store the best-performing result of each step
name_normalization_dict = {}  # store the name-normalization mapping

# =========== Utility functions ===========

def retry_execution(func, max_retries=3, **kwargs):
    """Retry logic for API calls."""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise
    return None

def setup_logging(output_dir, prefix):
    """Logging setup."""
    os.makedirs(output_dir, exist_ok=True)
    handler = logging.FileHandler(f"{output_dir}/{prefix}_log.txt")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def read_conversation_from_file(file_path):
    """Read conversation content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading conversation file: {e}")
        return None

def extract_prompts(file_path):
    try:
        notebook = nbformat.read(file_path, as_version=4)
        prompts = {
            # Basic step prompts with their techniques
            "Basic_ND": "",
            "Basic_ZS": "",
            "Basic_CoT": "",
            # Advanced step prompts with their techniques
            "Step2_CoT": "",
            "Step2_SR": "",
            "Step2_PD": "",
            "Step2_MoRE": "",
            "Step3_CoT": "",
            "Step3_SR": "",
            "Step3_PD": "",
            "Step3_MoRE": "",
            "Step4_CoT": "",
            "Step4_SR": "",
            "Step4_PD": "",
            "Step4_MoRE": ""
        }

        # Store data for debugging
        cell_info = []
        
        for idx, cell in enumerate(notebook.cells):
            cell_data = {
                "index": idx,
                "type": cell.cell_type
            }
            
            if cell.cell_type == 'markdown':
                content = ''.join(cell.source)
                cell_data["content_preview"] = content[:50] + "..." if len(content) > 50 else content
                
                if "Zero-Shot Prompting (No-Delimiters)" in content:
                    if idx + 1 < len(notebook.cells):
                        prompts["Basic_ND"] = extract_content(notebook.cells[idx + 1])
                        cell_data["matched"] = "Basic_ND"
                elif "Zero-Shot Prompting (Delimiters)" in content:
                    if idx + 1 < len(notebook.cells):
                        prompts["Basic_ZS"] = extract_content(notebook.cells[idx + 1])
                        cell_data["matched"] = "Basic_ZS"
                elif "Chain-of-Thought Prompting (CoT)" in content:
                    if idx + 1 < len(notebook.cells):
                        prompts["Basic_CoT"] = extract_content(notebook.cells[idx + 1])
                        cell_data["matched"] = "Basic_CoT"
            
            elif cell.cell_type == 'code':
                content = ''.join(cell.source)
                cell_data["content_preview"] = content[:50] + "..." if len(content) > 50 else content
                
                # Check for Step 2
                if "# Step 2:" in content:
                    if idx > 0:
                        prev_content = ''.join(notebook.cells[idx - 1].source)
                        
                        if "[Thought Generation] Chain-of-Thought (CoT)" in prev_content:
                            prompts["Step2_CoT"] = extract_content(cell)
                            cell_data["matched"] = "Step2_CoT"
                        elif "[Self-Criticism] Self-Refine Prompting" in prev_content:
                            prompts["Step2_SR"] = extract_content(cell)
                            cell_data["matched"] = "Step2_SR"
                        elif "[Decomposition] Prompt Decomposition" in prev_content:
                            prompts["Step2_PD"] = extract_content(cell)
                            cell_data["matched"] = "Step2_PD"
                        elif "[Ensembling] Mixture of Reasoning Experts (MoRE)" in prev_content:
                            prompts["Step2_MoRE"] = extract_content(cell)
                            cell_data["matched"] = "Step2_MoRE"
                
                # Check for Step 3
                elif "# Step 3:" in content:
                    if idx > 0:
                        prev_content = ''.join(notebook.cells[idx - 1].source)
                        
                        if "[Thought Generation] Chain-of-Thought (CoT)" in prev_content:
                            prompts["Step3_CoT"] = extract_content(cell)
                            cell_data["matched"] = "Step3_CoT"
                        elif "[Self-Criticism] Self-Refine Prompting" in prev_content:
                            prompts["Step3_SR"] = extract_content(cell)
                            cell_data["matched"] = "Step3_SR"
                        elif "[Decomposition] Prompt Decomposition" in prev_content:
                            prompts["Step3_PD"] = extract_content(cell)
                            cell_data["matched"] = "Step3_PD"
                        elif "[Ensembling] Mixture of Reasoning Experts (MoRE)" in prev_content:
                            prompts["Step3_MoRE"] = extract_content(cell)
                            cell_data["matched"] = "Step3_MoRE"
                
                # Check for Step 4
                elif "# Step 4:" in content:
                    if idx > 0:
                        prev_content = ''.join(notebook.cells[idx - 1].source)
                        
                        if "[Thought Generation] Chain-of-Thought (CoT)" in prev_content:
                            prompts["Step4_CoT"] = extract_content(cell)
                            cell_data["matched"] = "Step4_CoT"
                        elif "[Self-Criticism] Self-Refine Prompting" in prev_content:
                            prompts["Step4_SR"] = extract_content(cell)
                            cell_data["matched"] = "Step4_SR"
                        elif "[Decomposition] Prompt Decomposition" in prev_content:
                            prompts["Step4_PD"] = extract_content(cell)
                            cell_data["matched"] = "Step4_PD"
                        elif "[Ensembling] Mixture of Reasoning Experts (MoRE)" in prev_content:
                            prompts["Step4_MoRE"] = extract_content(cell)
                            cell_data["matched"] = "Step4_MoRE"
            
            if "matched" not in cell_data:
                cell_data["matched"] = None
                
            cell_info.append(cell_data)

        return prompts, cell_info
    
    except Exception as e:
        print(f"Error extracting prompts: {e}")
        return {}, []

def extract_content(cell):
    """Extract content from a cell."""
    if cell.cell_type == 'markdown':
        return ''.join(cell.source)
    elif cell.cell_type == 'code':
        return ''.join(cell.source)
    return ""

def normalize_name(name):
    """Name normalization (uses a global dictionary)."""
    global name_normalization_dict
    return name_normalization_dict.get(name.strip(), name.strip())

def standardize_name(name):
    """Name standardization."""
    return str(name).strip().lower()

def normalize_initial_result(result, name_mapping):
    """Normalize Step1 results."""
    normalized_result = copy.deepcopy(result)
    
    # Normalize participants
    if 'participants' in normalized_result:
        normalized_result['participants'] = [
            name_mapping.get(p.strip(), p.strip())
            for p in normalized_result['participants']
        ]
    
    # Normalize restaurants
    if 'restaurant_brands' in normalized_result:
        normalized_result['restaurant_brands'] = [
            name_mapping.get(r.strip(), r.strip())
            for r in normalized_result['restaurant_brands']
        ]
    
    # Normalize the final restaurant
    if 'final_restaurant' in normalized_result:
        normalized_result['final_restaurant'] = name_mapping.get(
            normalized_result['final_restaurant'].strip(),
            normalized_result['final_restaurant'].strip()
        )
    
    # Normalize suggestion/response tables
    if 'suggestion_table' in normalized_result:
        for item in normalized_result['suggestion_table']:
            if 'participant' in item:
                item['participant'] = name_mapping.get(
                    item['participant'].strip(),
                    item['participant'].strip()
                )
    
    if 'response_table' in normalized_result:
        for item in normalized_result['response_table']:
            if 'participant' in item:
                item['participant'] = name_mapping.get(
                    item['participant'].strip(),
                    item['participant'].strip()
                )
    
    return normalized_result

# =========== Analysis functions ===========

def analyze_step1(system_prompt, user_prompt, technique):
    """Run Step 1 analysis (participants, restaurants, suggestion/response traits)."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **get_api_kwargs(),
            tools=[FUNCTION_TOOLS["Steps1.1-1.2"]],
            tool_choice={"type": "function", "function": {"name": "analyze_step1"}}
        )
        function_call = response.choices[0].message.tool_calls[0]
        return json.loads(function_call.function.arguments)
    except Exception as e:
        logger.error(f"Error in analyze_step1: {e}")
        raise

def analyze_mentioned_table(system_prompt, user_prompt):
    """Run Step 2 analysis (first-mention table)."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **get_api_kwargs(),
            tools=[FUNCTION_TOOLS["Step2"]],
            tool_choice={"type": "function", "function": {"name": "output_mentioned_table"}}
        )
        function_call = response.choices[0].message.tool_calls[0]
        return json.loads(function_call.function.arguments)
    except Exception as e:
        logger.error(f"Error in analyze_mentioned_table: {e}")
        raise

def analyze_sentiment(system_prompt, user_prompt, technique):
    """Run Step 3 analysis (perception analysis)."""
    try:
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
        response = openai_client.chat.completions.create(
          model=MODEL,
          messages=messages,
          **get_api_kwargs(),
          tools=[FUNCTION_TOOLS["Step3"]],
          tool_choice={"type": "function", "function": {"name": "sentiment_analysis"}}
        )
        function_call = response.choices[0].message.tool_calls[0]
        args_str = function_call.function.arguments
        logger.info(f"Raw API response arguments: {args_str}")
        return json.loads(function_call.function.arguments)
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        raise
    
def analyze_preferences_constraints(system_prompt, user_prompt, technique):
    """Run Step 4 analysis (preferences and constraints)."""
    try:
        messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
       ]
        response = openai_client.chat.completions.create(
          model=MODEL,
          messages=messages,
          **get_api_kwargs(),
          tools=[FUNCTION_TOOLS["Step4"]],
          tool_choice={"type": "function", "function": {"name": "preference_constraint_analysis"}}
        )
        function_call = response.choices[0].message.tool_calls[0]
        return json.loads(function_call.function.arguments)
    except Exception as e:
        logger.error(f"Error in analyze_preferences_constraints: {e}")
        raise

# =========== Evaluation functions ===========

def calculate_f1_list(gold_list, pred_list):
    """F1 score computation for a list."""
    if not gold_list and not pred_list:
        return 1.0
    if not gold_list or not pred_list:
        return 0.0
    
    gold_set = set(map(str.strip, gold_list))
    pred_set = set(map(str.strip, pred_list))
   
    true_positives = len(gold_set & pred_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    if true_positives == 0:
        return 0.0
        
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_f1_table(gold_table, pred_table, key_columns):
    """F1 score computation for a table."""
    if not key_columns:
        return 0.0
        
    if not gold_table and not pred_table:
        return 1.0
    if not gold_table or not pred_table:
        return 0.0
        
    gold_rows = sorted([
        ' '.join(str(item.get(col, '')).strip() for col in key_columns) 
        for item in gold_table
    ])
    pred_rows = sorted([
        ' '.join(str(item.get(col, '')).strip() for col in key_columns) 
        for item in pred_table
    ])
    
    return calculate_f1_list(gold_rows, pred_rows)

def calculate_f1_scores(tp, fp, fn):
    """Basic F1 score computation."""
    if tp == 0:
        return 0.0
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)

def calculate_table_score_with_restaurants(gold_table, pred_table, restaurants_in_evaluation, key_columns, step=None):
    """Table evaluation (uses restaurant names extracted by the LLM)."""
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    
    # Convert gold and prediction data into pairs
    gold_pairs = {(item['participant'], normalize_name(item['restaurant'])): item 
                  for item in gold_table}
    pred_pairs = {(item['participant'], normalize_name(item['restaurant'])): item 
                  for item in pred_table}
    
    # Evaluate the gold cells
    for (participant, restaurant), gold_item in gold_pairs.items():
        if restaurant not in restaurants_in_evaluation:
            continue
        
        if (participant, restaurant) in pred_pairs:
            pred_item = pred_pairs[(participant, restaurant)]
            match = all(gold_item[col].strip() == pred_item[col].strip() 
                       for col in key_columns if col != 'restaurant')
            
            if match:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            false_negatives += 1
    
    # Evaluate incorrect predictions
    for (participant, restaurant) in pred_pairs:
        if restaurant not in restaurants_in_evaluation:
            false_positives += 1
        elif (participant, restaurant) not in gold_pairs:
            false_positives += 1
    
    # Compute F1 score
    return calculate_f1_scores(true_positives, false_positives, false_negatives)

def extract_restaurants_from_results(step_result, gold_answer, step):
    """Function to evaluate based on the restaurant names extracted by the LLM."""
    global best_results  
    if 'step1' in best_results and 'restaurant_brands' in best_results['step1']:
        restaurants_in_evaluation = set(best_results['step1']['restaurant_brands'])
    else:
        restaurants_in_evaluation = set(gold_answer.get('restaurant_brands', []))
    return restaurants_in_evaluation

def parse_factors(factor_str):
    """Convert a Factor string into a set."""
    if not factor_str or factor_str.strip().lower() == 'none':
        return set()
    return set(f.strip() for f in factor_str.split(',') if f.strip())

def calculate_similarity_by_metric(set1, set2, metric_type='jaccard'):
    """Compute similarity between sets (Jaccard or Factor F1)."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    
    if metric_type == 'jaccard':
        union = len(set1.union(set2))
        return intersection / union
    else:  # factor_f1
        precision = intersection / len(set2) if set2 else 0.0
        recall = intersection / len(set1) if set1 else 0.0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def calculate_table_similarity_with_restaurants(gold_table, pred_table, restaurants_in_evaluation, factor_column, metric_type='jaccard'):
    """Compute similarity of table elements (for Factor analysis)."""
    total_score = 0
    total_pairs = 0

    # Convert gold and prediction data into pairs
    gold_pairs = {(item['participant'], normalize_name(item['restaurant'])): parse_factors(item[factor_column]) 
                  for item in gold_table}
    pred_pairs = {(item['participant'], normalize_name(item['restaurant'])): parse_factors(item[factor_column]) 
                  for item in pred_table}
    
    # Evaluate the gold cells
    for (participant, restaurant), gold_factors in gold_pairs.items():
        if restaurant not in restaurants_in_evaluation:
            continue
        
        if (participant, restaurant) in pred_pairs:
            pred_factors = pred_pairs[(participant, restaurant)]
            similarity = calculate_similarity_by_metric(gold_factors, pred_factors, metric_type)
            total_score += similarity
            total_pairs += 1
    
    # Return the mean similarity score
    return total_score / total_pairs if total_pairs > 0 else 0.0

def analyze_step_performance(gold_answer, step_result, step):
    """Analyze the performance of each detailed evaluation item per Step."""
    performance = {}
    
#     if step == "Steps1.1-1.2":
#         # Per-step performance evaluation for Steps1.1-1.2
#         step11_perf = analyze_step_performance(gold_answer, step_result, "Step1.1")
#         step12_perf = analyze_step_performance(gold_answer, step_result, "Step1.2")
        
#         # Individual score of each Step
#         performance['step11_score'] = step11_perf['step11_score']
#         performance['step12_score'] = step12_perf['step12_score']
        
#         # Steps1 overall score
#         performance['Step1_score'] = (performance['step11_score'] + performance['step12_score']) / 2

    if step == "Steps1.1-1.2":
        # Per-step performance evaluation for Steps1.1-1.2
        step11_perf = analyze_step_performance(gold_answer, step_result, "Step1.1")
        step12_perf = analyze_step_performance(gold_answer, step_result, "Step1.2")

        # Copy each Step's individual score and all detailed metrics
        performance['step11_score'] = step11_perf['step11_score']
        performance['step12_score'] = step12_perf['step12_score']

        # Copy the individual metrics as well
        performance['f1_participants'] = step11_perf['f1_participants']
        performance['f1_restaurants'] = step11_perf['f1_restaurants']
        performance['final_restaurant_match'] = step11_perf['final_restaurant_match']
        performance['f1_suggestion'] = step12_perf['f1_suggestion']
        performance['f1_response'] = step12_perf['f1_response']

        # Steps1 overall score
        performance['Step1_score'] = (performance['step11_score'] + performance['step12_score']) / 2

    elif step == "Step1.1":
        # Step1.1 Score = (F1_participant + F1_restaurant + Final_Restaurant_Binary) / 3
        f1_participants = calculate_f1_list(
            gold_answer['participants'], 
            step_result.get('participants', [])
        )
        f1_restaurants = calculate_f1_list(
            gold_answer['restaurant_brands'], 
            step_result.get('restaurant_brands', [])
        )
        final_restaurant_match = 1.0 if gold_answer['final_restaurant'].strip() == step_result.get('final_restaurant', '').strip() else 0.0
        
        performance['f1_participants'] = f1_participants
        performance['f1_restaurants'] = f1_restaurants
        performance['final_restaurant_match'] = final_restaurant_match
        performance['step11_score'] = (f1_participants + f1_restaurants + final_restaurant_match) / 3

    elif step == "Step1.2":
        # Step1.2 Score = (F1_suggestion + F1_response) / 2
        f1_suggestion = calculate_f1_table(
            gold_answer['suggestion_table'], 
            step_result.get('suggestion_table', []),
            ['participant', 'suggestion_type']
        )
        f1_response = calculate_f1_table(
            gold_answer['response_table'], 
            step_result.get('response_table', []),
            ['participant', 'response_type']
        )
        
        performance['f1_suggestion'] = f1_suggestion
        performance['f1_response'] = f1_response
        performance['step12_score'] = (f1_suggestion + f1_response) / 2

    elif step == "Step2":
        # Step2 Score = F1_Mentioned only
        restaurants_in_evaluation = extract_restaurants_from_results(step_result, gold_answer, step)
        mentioned_only = lambda table: [item for item in table if item['mention'] == 'Mentioned']
        
        f1_mentioned = calculate_table_score_with_restaurants(
            mentioned_only(gold_answer['mentioned_table']),
            mentioned_only(step_result['mentioned_table']),
            restaurants_in_evaluation,
            ['participant', 'restaurant', 'mention']
        )
        
        performance['F1_Mentioned'] = f1_mentioned
        performance['step2_score'] = f1_mentioned

    elif step == "Step3":
        # Step3 Score = F1_total
        restaurants_in_evaluation = extract_restaurants_from_results(step_result, gold_answer, step)
        
        # Convert gold_answer's perception_table
        gold_sentiment_table = []
        for item in gold_answer.get('perception_table', []):
            # Convert the perception field to sentiment
            gold_item = {
                'participant': item.get('participant', ''),
                'restaurant': item.get('restaurant', ''),
                'sentiment': item.get('perception', '')  # use perception as sentiment
            }
            gold_sentiment_table.append(gold_item)
        
        f1_total = calculate_table_score_with_restaurants(
            gold_sentiment_table,
            step_result['sentiment_table'],
            restaurants_in_evaluation,
            ['participant', 'restaurant', 'sentiment']  # both tables now have a sentiment field
        )
        
        performance['F1_total'] = f1_total
        performance['step3_score'] = f1_total

    elif step == "Step4":
        # Step4 uses both JS Score and Factor F1
        restaurants_in_evaluation = extract_restaurants_from_results(step_result, gold_answer, step)
        
        preference_js = calculate_table_similarity_with_restaurants(
            gold_answer['preference_table'],
            step_result['preference_table'],
            restaurants_in_evaluation,
            'preferences',
            'jaccard'
        )
        
        preference_f1 = calculate_table_similarity_with_restaurants(
            gold_answer['preference_table'],
            step_result['preference_table'],
            restaurants_in_evaluation,
            'preferences',
            'factor_f1'
        )
        
        constraint_js = calculate_table_similarity_with_restaurants(
            gold_answer['constraint_table'],
            step_result['constraint_table'],
            restaurants_in_evaluation,
            'constraints',
            'jaccard'
        )
        
        constraint_f1 = calculate_table_similarity_with_restaurants(
            gold_answer['constraint_table'],
            step_result['constraint_table'],
            restaurants_in_evaluation,
            'constraints',
            'factor_f1'
        )
        
        performance['Total_JS_Score'] = (preference_js + constraint_js) / 2
        performance['Total_Factor_F1'] = (preference_f1 + constraint_f1) / 2
        performance['preference_js'] = preference_js
        performance['preference_f1'] = preference_f1
        performance['constraint_js'] = constraint_js
        performance['constraint_f1'] = constraint_f1

    return performance

# =========== Utility functions (additional) ===========

def collect_name_variations(results):
    """Collect name variations from the initial results."""
    participants = set()
    restaurants = set()
    
    for result in results:
        raw_result = result.get('Raw_Result', {})
        participants.update(raw_result.get('participants', []))
        restaurants.update(raw_result.get('restaurant_brands', []))
    
    return sorted(list(participants)), sorted(list(restaurants))

def create_variations_excel(output_dir, participants, restaurants):
    """Create an Excel file to record name variations."""
    excel_path = os.path.join(output_dir, "name_variations.xlsx")
    
    # Create the Excel file
    with pd.ExcelWriter(excel_path) as writer:
        # Participant sheet
        df_participants = pd.DataFrame({
            'Original Name': participants,
            'Standard Name': participants  # use the original name as default
        })
        df_participants.to_excel(writer, sheet_name='Participants', index=False)
        
        # Restaurant sheet
        df_restaurants = pd.DataFrame({
            'Original Name': restaurants,
            'Standard Name': restaurants  # use the original name as default
        })
        df_restaurants.to_excel(writer, sheet_name='Restaurants', index=False)
    
    return excel_path

def read_mapping_from_excel(excel_path):
    """Read the name mapping from an Excel file."""
    mapping_dict = {}
    
    # Read participant mapping
    df_participants = pd.read_excel(excel_path, sheet_name='Participants')
    for _, row in df_participants.iterrows():
        mapping_dict[row['Original Name']] = row['Standard Name']
    
    # Read restaurant mapping
    df_restaurants = pd.read_excel(excel_path, sheet_name='Restaurants')
    for _, row in df_restaurants.iterrows():
        mapping_dict[row['Original Name']] = row['Standard Name']
    
    return mapping_dict

def load_gold_answers(step, gold_dir):
    """Load per-step gold answers."""
    try:
        if step == "Step1":
            # Step1 loads two files (1.1 and 1.2)
            with open(os.path.join(gold_dir, "step1_1_gold.json"), 'r', encoding='utf-8') as f1:
                gold1 = json.load(f1)
            with open(os.path.join(gold_dir, "step1_2_gold.json"), 'r', encoding='utf-8') as f2:
                gold2 = json.load(f2)
            # Merge the two results
            gold = {**gold1, **gold2}
            return gold
        else:
            # Other steps load a single file
            file_name = f"step{step[-1]}_gold.json"  # "Step2" -> "step2_gold.json"
            with open(os.path.join(gold_dir, file_name), 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading gold answers for {step}: {e}")
        return None

def calculate_f1_for_step(step, gold_answers, result):
    """Per-step F1 score computation."""
    if step == "Steps1.1-1.2":
        # Average of Step1.1 + Step1.2
        perf = analyze_step_performance(gold_answers, result, "Steps1.1-1.2")
        return perf.get('Step1_score', 0)
    elif step == "Step2":
        # Step2 score
        perf = analyze_step_performance(gold_answers, result, step)
        return perf.get('step2_score', 0)
    elif step == "Step3":
        # Step3 score
        perf = analyze_step_performance(gold_answers, result, step)
        return perf.get('step3_score', 0)
    elif step == "Step4":
        # Step4 score (JS + F1) / 2
        perf = analyze_step_performance(gold_answers, result, step)
        js_score = perf.get('Total_JS_Score', 0)
        f1_score = perf.get('Total_Factor_F1', 0)
        return (js_score + f1_score) / 2
    return 0

def select_best_technique(step_performance, step):
    """Select the best-performing technique."""
    if not step_performance:
        return None, None
    
    # Sort by average score
    sorted_techniques = sorted(step_performance, key=lambda x: x[1], reverse=True)
    best_technique, best_avg, performance_list = sorted_techniques[0]
    
    # Select the highest-scoring iteration within that technique
    best_result = max(performance_list, key=lambda x: x['Score'])
    
    return best_technique, best_result

def select_best_technique_step4(step_performance):
    """Select the best-performing technique for Step4 (JS and F1 separated)."""
    results = {
        'best_js_technique': "",
        'best_js_score': 0,
        'best_js': None,
        'best_f1_technique': "",
        'best_f1_score': 0,
        'best_f1': None
    }
    
    for technique, scores, performance_list in step_performance:
        # Find the best result by JS
        js_scores = [(r, r['Score']['JS']) for r in performance_list]
        best_js = max(js_scores, key=lambda x: x[1])
        
        if best_js[1] > results['best_js_score']:
            results['best_js_score'] = best_js[1]
            results['best_js'] = best_js[0]
            results['best_js_technique'] = technique
        
        # Find the best result by F1
        f1_scores = [(r, r['Score']['F1']) for r in performance_list]
        best_f1 = max(f1_scores, key=lambda x: x[1])
        
        if best_f1[1] > results['best_f1_score']:
            results['best_f1_score'] = best_f1[1]
            results['best_f1'] = best_f1[0]
            results['best_f1_technique'] = technique
    
    return results

def create_system_prompt(base_prompt, conversation, previous_results):
    """Build a system prompt that includes previous results."""
    prompt = f"{base_prompt}\n\nConversation:\n{conversation}\n\n"
    
    # Append previous-step results
    if previous_results:
        prompt += "Previous Analysis Results:\n"
        for step, result in previous_results.items():
            prompt += f"--- {step.upper()} RESULTS ---\n{result}\n\n"
    
    return prompt

def format_previous_result(step, result):
    """Format previous-step results."""
    if step == "step1":
        # Format Step1 results
        formatted = f"Participants: {', '.join(result.get('participants', []))}\n"
        formatted += f"Restaurants: {', '.join(result.get('restaurant_brands', []))}\n"
        formatted += f"Final Restaurant: {result.get('final_restaurant', '')}\n"
        
        # Suggestion table
        formatted += "\nSuggestion Table:\n"
        for item in result.get('suggestion_table', []):
            formatted += f"- {item.get('participant', '')}: {item.get('suggestion_type', '')}\n"
        
        # Response table
        formatted += "\nResponse Table:\n"
        for item in result.get('response_table', []):
            formatted += f"- {item.get('participant', '')}: {item.get('response_type', '')}\n"
        
        return formatted
    
    elif step == "step2":
        # Format Step2 results
        formatted = "Mention Table:\n"
        for item in result.get('mentioned_table', []):
            if item.get('mention') == 'Mentioned':
                formatted += f"- {item.get('participant', '')} mentioned {item.get('restaurant', '')}\n"
        
        return formatted
    
    elif step == "step3":
        # Format Step3 results
        formatted = "Sentiment Table:\n"
        for item in result.get('sentiment_table', []):
            formatted += f"- {item.get('participant', '')} feels {item.get('sentiment', '')} about {item.get('restaurant', '')}\n"
        
        return formatted
    
    elif step == "step4":
        # Format Step4 results (choose between JS and F1)
        js_result = result.get('js', {})
        
        formatted = "Preferences:\n"
        for item in js_result.get('preference_table', []):
            factors = item.get('preferences', 'None')
            formatted += f"- {item.get('participant', '')} prefers {item.get('restaurant', '')} for: {factors}\n"
        
        formatted += "\nConstraints:\n"
        for item in js_result.get('constraint_table', []):
            factors = item.get('constraints', 'None')
            formatted += f"- {item.get('participant', '')} has concerns about {item.get('restaurant', '')} for: {factors}\n"
        
        return formatted
    
    return ""

def create_step_summary(step, step_performance, best_result, P_star=None, R_star=None):
    """Generate a per-step result summary."""
    summary = f"=== {step} Summary ===\n\n"
    
    # Per-technique average score
    summary += "Technique Performance:\n"
    for technique, avg_score, _ in step_performance:
        if isinstance(avg_score, dict):  # for Step4
            summary += f"- {technique}: JS={avg_score.get('JS', 0):.4f}, F1={avg_score.get('F1', 0):.4f}\n"
        else:
            summary += f"- {technique}: {avg_score:.4f}\n"
    
    summary += "\nBest Result:\n"
    if step == "Step4":
        # Step4 handles JS and F1 separately
        js_score = best_result.get('js', {})
        f1_score = best_result.get('f1', {})
        
        summary += f"- Best JS Score: {best_result.get('best_js_score', 0):.4f} (Technique: {best_result.get('best_js_technique', '')})\n"
        summary += f"- Best F1 Score: {best_result.get('best_f1_score', 0):.4f} (Technique: {best_result.get('best_f1_technique', '')})\n"
    else:
        # Other steps
        summary += f"- Score: {best_result.get('Score', 0):.4f}\n"
        summary += f"- Technique: {best_result.get('Technique', '')}\n"
        summary += f"- Iteration: {best_result.get('Iteration', '')}\n"
    
    # Add P* and R* info (if available)
    if P_star and R_star:
        summary += f"\nValid Space:\n"
        summary += f"- P*: {', '.join(P_star)}\n"
        summary += f"- R*: {', '.join(R_star)}\n"
    
    return summary

def save_step_summary(summary, step, txt_name, output_dir):
    """Save the per-step summary."""
    summary_path = os.path.join(output_dir, f"{txt_name}_{step}_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        return True
    except Exception as e:
        logger.error(f"Error saving summary to {summary_path}: {e}")
        return False

def save_iteration_result(txt_name, step, technique, iteration, result, score, detailed_performance, output_dir):
    """Save iteration results."""
    # Build the result object
    result_obj = {
        "Result": result,
        "Score": score,
        "Detailed_Performance": detailed_performance
    }
    
    # Build the file path
    file_path = os.path.join(output_dir, f"{txt_name}_{step}_{technique}_{iteration}.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_obj, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving iteration result to {file_path}: {e}")
        return False

def save_checkpoint(results, txt_name, output_dir):
    """Save checkpoint."""
    checkpoint_path = os.path.join(output_dir, f"{txt_name}_checkpoint.json")
    try:
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
        return False

def load_checkpoint(txt_name, output_dir):
    """Load checkpoint."""
    checkpoint_path = os.path.join(output_dir, f"{txt_name}_checkpoint.json")
    try:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
    return None

def save_final_results(results, txt_name, output_dir):
    """Save final results."""
    final_path = os.path.join(output_dir, f"{txt_name}_final_results.json")
    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving final results to {final_path}: {e}")
        return False

def save_json_result(data, filepath):
    """Save data to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON result to {filepath}: {e}")
        return False

# ============ Metrics Saving (Cell 5) ============
def save_iteration_metrics(txt_name, step, technique, iterations_data, output_dir):
    """Save the evaluation-metric values of each iteration to a CSV file."""
    # Prepare evaluation-metric data
    metrics_data = []
    
    # Define evaluation metrics for each step
    if step == "Step1":
        metrics = ['f1_participants', 'f1_restaurants', 'final_restaurant_match', 'step11_score', 
                   'f1_suggestion', 'f1_response', 'step12_score', 'Step1_score']
    elif step == "Step2":
        metrics = ['F1_Mentioned', 'step2_score']
    elif step == "Step3":
        metrics = ['F1_total', 'step3_score']
    elif step == "Step4":
        metrics = ['preference_js', 'preference_f1', 'constraint_js', 'constraint_f1', 
                   'Total_JS_Score', 'Total_Factor_F1']
    
    # Extract metric values from each iteration
    for result in iterations_data:
        iteration = result.get('Iteration', 0)
        
        # Important fix: change Detailed_Performance to be directly accessible
        detailed_perf = result.get('Detailed_Performance', {})
        
        row = {'Iteration': iteration}
        for metric in metrics:
            # Access metric values (now directly accessible)
            row[metric] = detailed_perf.get(metric, 0)
        
        metrics_data.append(row)
    
    # Build and save DataFrame
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(output_dir, f"{txt_name}_{step}_{technique}_metrics.csv")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved iteration metrics to {csv_path}")
    
    return csv_path

def save_technique_summary_metrics(txt_name, step, step_performance, output_dir):
    """Save the per-technique evaluation-metric summary (mean, std, max, min) to a CSV file."""
    # Define evaluation metrics for each step
    if step == "Step1":
        metrics = ['f1_participants', 'f1_restaurants', 'final_restaurant_match', 'step11_score', 
                   'f1_suggestion', 'f1_response', 'step12_score', 'Step1_score']
    elif step == "Step2":
        metrics = ['F1_Mentioned', 'step2_score']
    elif step == "Step3":
        metrics = ['F1_total', 'step3_score']
    elif step == "Step4":
        metrics = ['preference_js', 'preference_f1', 'constraint_js', 'constraint_f1', 
                   'Total_JS_Score', 'Total_Factor_F1']
    
    # List for storing results
    summary_data = []
    
    # Compute metrics for each technique
    for technique, avg_score, performance_list in step_performance:
        technique_stats = {'Technique': technique}
        
        # Compute statistics for each metric
        for metric in metrics:
            # Important fix: access metric values directly from Detailed_Performance
            values = [r.get('Detailed_Performance', {}).get(metric, 0) for r in performance_list]
            
            if values:
                technique_stats[f"{metric}_Mean"] = np.mean(values)
                technique_stats[f"{metric}_Std"] = np.std(values)
                technique_stats[f"{metric}_Max"] = np.max(values)
                technique_stats[f"{metric}_Min"] = np.min(values)
            else:
                technique_stats[f"{metric}_Mean"] = 0
                technique_stats[f"{metric}_Std"] = 0
                technique_stats[f"{metric}_Max"] = 0
                technique_stats[f"{metric}_Min"] = 0
        
        summary_data.append(technique_stats)
    
    # Build and save DataFrame
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f"{txt_name}_{step}_technique_summary.csv")
    
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved technique summary metrics to {csv_path}")
    
    return csv_path

# ============ Fuzzy Matching (Cell 7) ============
import difflib

def fuzzy_match(name, candidates, threshold=0.6):
    """Automated fuzzy matching for name normalization."""
    name_clean = name.strip()
    
    # 1) Exact match
    for c in candidates:
        if name_clean == c.strip():
            return c.strip(), 1.0
    
    # 2) Whitespace/bracket normalization
    import re
    def norm(s):
        return re.sub(r'\\s+', '', s.strip()).lower()
    for c in candidates:
        if norm(name_clean) == norm(c):
            return c.strip(), 0.99
    
    # 3) difflib SequenceMatcher
    best_match, best_score = None, 0
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, name_clean.lower(), c.strip().lower()).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = c.strip()
    if best_score >= threshold:
        return best_match, best_score
    
    # 4) Substring matching
    for c in candidates:
        c_clean = c.strip()
        if name_clean in c_clean or c_clean in name_clean:
            return c_clean, 0.8
    
    return best_match, best_score


def build_auto_name_mapping(initial_results, gold_dir, log_name):
    """Build name mapping automatically using fuzzy matching against gold standard."""
    import os
    
    # Collect all name variations from LLM outputs
    all_participants, all_restaurants = set(), set()
    for result in initial_results:
        raw = result.get('Raw_Result', {})
        all_participants.update(raw.get('participants', []))
        all_restaurants.update(raw.get('restaurant_brands', []))
    
    # Load gold standard names
    gold_s11 = {}
    try:
        with open(os.path.join(gold_dir, 'step1_1_gold.json'), 'r', encoding='utf-8') as f:
            gold_s11 = json.load(f)
    except:
        pass
    
    gold_participants = gold_s11.get('participants', [])
    gold_restaurants = gold_s11.get('restaurant_brands', [])
    
    mapping = {}
    for p in all_participants:
        match, score = fuzzy_match(p.strip(), gold_participants)
        mapping[p.strip()] = match if match and score >= 0.6 else p.strip()
    
    for r in all_restaurants:
        match, score = fuzzy_match(r.strip(), gold_restaurants)
        mapping[r.strip()] = match if match and score >= 0.6 else r.strip()
    
    return mapping

print("Fuzzy matching functions defined")

# ============ Run Analysis (Cell 9) ============
def run_analysis(conversation_file, prompt_file, output_dir, gold_dir, num_iterations=5, start_step=None):
    """Analysis-run function that can specify a starting Step."""
    try:
        # Create the required directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the file name
        txt_name = os.path.splitext(os.path.basename(conversation_file))[0]
        
        # Logging setup
        logger = setup_logging(output_dir, txt_name)
        logger.info(f"Starting analysis for {txt_name}")
        
        # Initialize the name-normalization dictionary and result store
        global name_normalization_dict, best_results
        name_normalization_dict = {}
        best_results = {}
        
        # Load conversation and prompt
        conversation = read_conversation_from_file(conversation_file)
        if not conversation:
            logger.error("Failed to load conversation")
            return
        
        PROMPT_TEMPLATES, cell_info = extract_prompts(prompt_file)
        if not PROMPT_TEMPLATES:
            logger.error("Failed to load prompt templates")
            return
        
        # Run Step1 (if needed)
        if not start_step or start_step == "Step1":
            logger.info("\nProcessing Step1")
            
            # Load Step1 gold answers
            gold_answers = load_gold_answers("Step1", gold_dir)
            if not gold_answers:
                logger.error("Failed to load gold answers for Step1")
                return
                
            # List for storing initial results
            initial_results = []
            
            # Build the system prompt
            system_prompt = f"{COMMON_SYSTEM_PROMPT}\n\nConversation:\n{conversation}"
            
            # Run for each technique (ND, ZS, CoT)
            for technique in ["ND", "ZS", "CoT"]:
                logger.info(f"Running Step1 with {technique} technique")
                for iteration in range(1, num_iterations + 1):
                    try:
                        # Run Step1 analysis
                        result = retry_execution(
                            analyze_step1,
                            max_retries=3,
                            system_prompt=system_prompt,
                            user_prompt=PROMPT_TEMPLATES.get(f"Basic_{technique}"),
                            technique=technique
                        )
                        
                        if result:
                            # Save results
                            result_entry = {
                                'Step': 'Step1',
                                'Technique': technique,
                                'Iteration': iteration,
                                'Raw_Result': result
                            }
                            initial_results.append(result_entry)
                            
                            # Save initial results - saved per technique
                            save_json_result(
                                result, 
                                os.path.join(output_dir, f"Step1_{technique}_initial_{iteration}.json")
                            )
                    except Exception as e:
                        logger.error(f"Error in Step1 iteration {iteration} with {technique}: {e}")
                        continue
            
            # Save all initial results to a single file
            save_json_result(initial_results, os.path.join(output_dir, "Step1_initial.json"))
            
            # Perform name normalization
            logger.info("Collecting name variations from initial results...")
            participants, restaurants = collect_name_variations(initial_results)
            
            # Create an Excel file for name mapping
            # Build name mapping via automatic fuzzy matching
            logger.info("Building automated name mapping via fuzzy matching...")
            name_normalization_dict = build_auto_name_mapping(initial_results, gold_dir, txt_name)
            logger.info(f"Auto-mapped {len(name_normalization_dict)} names")
            
            # Save reference Excel (for inspection)
            excel_path = create_variations_excel(output_dir, participants, restaurants)
            logger.info(f"Reference Excel saved at {excel_path}")
            
            # Save normalized results
            normalized_results = []
            
            # Evaluate performance on normalized results
            step_performance = []
            for technique in ["ND", "ZS", "CoT"]:
                performance_list = []
                technique_results = [r for r in initial_results if r['Technique'] == technique]
                
                for result in technique_results:
                    # Apply name normalization
                    normalized_result = normalize_initial_result(
                        result['Raw_Result'], 
                        name_normalization_dict
                    )
                    
                    # Save normalized results
                    normalized_entry = {
                        'Step': 'Step1',
                        'Technique': technique,
                        'Iteration': result['Iteration'],
                        'Raw_Result': normalized_result
                    }
                    normalized_results.append(normalized_entry)
                    
                    # Save individual normalized results
                    save_json_result(
                        normalized_result,
                        os.path.join(output_dir, f"Step1_{technique}_normalized_{result['Iteration']}.json")
                    )
                    
                    # Evaluate Step1 performance
                    score = calculate_f1_for_step("Steps1.1-1.2", gold_answers, normalized_result)
                    detailed_performance = analyze_step_performance(gold_answers, normalized_result, "Steps1.1-1.2")

                    result_entry = {
                        'Step': 'Step1',
                        'Technique': technique,
                        'Iteration': result['Iteration'],
                        'Score': score,
                        'Raw_Result': normalized_result,
                        'Detailed_Performance': detailed_performance
                    }
                    performance_list.append(result_entry)
                
                if performance_list:
                    # Compute per-technique average score
                    avg_score = sum(r['Score'] for r in performance_list) / len(performance_list)
                    step_performance.append((technique, avg_score, performance_list))
            
            # Select the best-performing technique
            best_technique, best_result = select_best_technique(step_performance, "Step1")
            best_results['step1'] = best_result['Raw_Result']
            
            # Generate and save the result summary
            summary = create_step_summary("Step1", step_performance, best_result)
            save_step_summary(summary, "Step1", txt_name, output_dir)
            
            # Set P* and R* (from the best-performing result)
            P_star = best_result['Raw_Result'].get('participants', [])
            R_star = best_result['Raw_Result'].get('restaurant_brands', [])
            logger.info(f"Selected P*: {P_star}")
            logger.info(f"Selected R*: {R_star}")
            
            # Save all normalized results to a single file
            save_json_result(normalized_results, os.path.join(output_dir, "Step1_mapping.json"))
            
            # Save per-technique metrics for Step1
            for technique, _, performance_list in step_performance:
                save_iteration_metrics(txt_name, "Step1", technique, performance_list, output_dir)

            # Save per-technique summary metrics for Step1
            save_technique_summary_metrics(txt_name, "Step1", step_performance, output_dir)
            
        else:
            # When skipping Step1, load previous results from the checkpoint
            logger.info(f"Skipping Step1, starting from {start_step}")
            checkpoint = load_checkpoint(txt_name, output_dir)
            
            if checkpoint and 'step1' in checkpoint:
                best_results = checkpoint
                P_star = best_results['step1'].get('participants', [])
                R_star = best_results['step1'].get('restaurant_brands', [])
                logger.info(f"Loaded P*: {P_star}")
                logger.info(f"Loaded R*: {R_star}")
            else:
                logger.error("No checkpoint found for Step1. Cannot continue.")
                return
        
        # Run subsequent steps
        next_steps = ["Step2", "Step3", "Step4"]
        if start_step:
            try:
                start_idx = next_steps.index(start_step)
                next_steps = next_steps[start_idx:]
            except ValueError:
                if start_step != "Step1":
                    logger.error(f"Invalid start step: {start_step}")
                    return
        
        # Run each step sequentially
        for step in next_steps:
            logger.info(f"\nProcessing {step}")
            
            # Load gold answers
            gold_answers = load_gold_answers(step, gold_dir)
            if not gold_answers:
                logger.error(f"Failed to load gold answers for {step}")
                continue
            
            # Build a system prompt including previous-step results
            previous_results = {k: format_previous_result(k, v) for k, v in best_results.items()}
            system_prompt = create_system_prompt(
                COMMON_SYSTEM_PROMPT,
                conversation,
                previous_results
            )
            
            # Run experiments per technique (CoT, PD, SR, MoRE)
            step_performance = []
            for technique in ["CoT", "PD", "SR", "MoRE"]:
                logger.info(f"Running {step} with {technique} technique")
                performance_list = []
                
                for iteration in range(1, num_iterations + 1):
                    try:
                        # Call the per-step analysis function
                        if step == "Step2":
                            user_prompt = PROMPT_TEMPLATES.get(f"{step}_{technique}")
                            result = retry_execution(
                                analyze_mentioned_table,
                                max_retries=3,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt
                            )
                        elif step == "Step3":
                            user_prompt = PROMPT_TEMPLATES.get(f"{step}_{technique}")
                            result = retry_execution(
                                analyze_sentiment,
                                max_retries=3,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                technique=technique
                            )
                        elif step == "Step4":
                            user_prompt = PROMPT_TEMPLATES.get(f"{step}_{technique}")
                            result = retry_execution(
                                analyze_preferences_constraints,
                                max_retries=3,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                technique=technique
                            )
                        
                        if result:
                            # Evaluate performance
                            detailed_performance = analyze_step_performance(
                                gold_answers, result, step
                            )
                            
                            # Compute per-step score
                            if step == "Step4":
                                js_score = detailed_performance.get('Total_JS_Score', 0)
                                f1_score = detailed_performance.get('Total_Factor_F1', 0)
                                score = {'JS': js_score, 'F1': f1_score}
                            else:
                                score = detailed_performance.get(f'step{step[-1]}_score', 0)
                            
                            # Save results
                            result_entry = {
                                'Step': step,
                                'Technique': technique,
                                'Iteration': iteration,
                                'Score': score,
                                'Raw_Result': result,
                                'Detailed_Performance': detailed_performance
                            }
                            performance_list.append(result_entry)
                            
                            # Save iteration results
                            save_iteration_result(
                                txt_name, step, technique, iteration,
                                result, score, detailed_performance, output_dir
                            )
                    except Exception as e:
                        logger.error(f"Error in {step} iteration {iteration} with {technique}: {e}")
                        continue
                
                if performance_list:
                    # Step4 uses two metrics (JS, F1), so handle it differently
                    if step == "Step4":
                        avg_js = sum(r['Score']['JS'] for r in performance_list) / len(performance_list)
                        avg_f1 = sum(r['Score']['F1'] for r in performance_list) / len(performance_list)
                        step_performance.append((technique, {'JS': avg_js, 'F1': avg_f1}, performance_list))
                    else:
                        avg_score = sum(r['Score'] for r in performance_list) / len(performance_list)
                        step_performance.append((technique, avg_score, performance_list))
            
            # Select the best-performing result
            if step == "Step4":
                # Step4 selects the best result for JS and F1 separately
                step_results = select_best_technique_step4(step_performance)
                best_results[step.lower()] = {
                    'js': step_results['best_js']['Raw_Result'],
                    'f1': step_results['best_f1']['Raw_Result']
                }

                # Print results
                logger.info(f"\nStep4 Results:")
                logger.info(f"Best JS Score: {step_results['best_js_score']:.4f} (Technique: {step_results['best_js_technique']})")
                logger.info(f"Best Factor F1: {step_results['best_f1_score']:.4f} (Technique: {step_results['best_f1_technique']})")
            else:
                best_technique, best_result = select_best_technique(step_performance, step)
                best_results[step.lower()] = best_result['Raw_Result']
                
                # Print results
                logger.info(f"\nBest technique for {step}: {best_technique}")
                logger.info(f"Best score: {best_result['Score']:.4f}")
            
            # Save per-technique metrics for this step
            for technique, _, performance_list in step_performance:
                save_iteration_metrics(txt_name, step, technique, performance_list, output_dir)

            # Save per-technique summary metrics for this step
            save_technique_summary_metrics(txt_name, step, step_performance, output_dir)
            
            
            # Generate and save the result summary
            # Revised version:
            if step == "Step4":
                # For Step4, pass in the appropriate form
                summary = create_step_summary(step, step_performance, 
                                             {"best_js_score": step_results['best_js_score'],
                                              "best_js_technique": step_results['best_js_technique'],
                                              "best_f1_score": step_results['best_f1_score'],
                                              "best_f1_technique": step_results['best_f1_technique'],
                                              "js": step_results['best_js']['Raw_Result'],
                                              "f1": step_results['best_f1']['Raw_Result']}, P_star, R_star)
            else:
                # For other steps, pass as before
                summary = create_step_summary(step, step_performance, 
                                             best_result, P_star, R_star)
            save_step_summary(summary, step, txt_name, output_dir)
            
            # Save checkpoint
            save_checkpoint(best_results, txt_name, output_dir)
        
        # Save final results
        save_final_results(best_results, txt_name, output_dir)
        logger.info("\nAnalysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.info("Use run_analysis with start_step parameter to continue from a specific step")

# ============ Init & Main ============
def init(model_name, temperature):
    """Initialize module-level config for this process."""
    global MODEL, MODEL_NAME, TEMPERATURE
    MODEL = model_name
    MODEL_NAME = model_name
    TEMPERATURE = temperature


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("conv_file")
    _parser.add_argument("prompt_file")
    _parser.add_argument("output_dir")
    _parser.add_argument("gold_dir")
    _parser.add_argument("num_iter", type=int)
    _parser.add_argument("model_name")
    _parser.add_argument("temperature")  # "None" or float
    _args = _parser.parse_args()
    
    _temp = None if _args.temperature == "None" else float(_args.temperature)
    init(_args.model_name, _temp)
    
    run_analysis(_args.conv_file, _args.prompt_file, _args.output_dir, _args.gold_dir, _args.num_iter)