import os
import json
import re
from typing import Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Import from our local schemas file
from src.schemas import ProposalState, Config

# Load environment variables (API Key)
load_dotenv()

# --- Helper Functions ---
def extract_json_robust(text: str) -> dict:
    """Extracts JSON from LLM output, handling Markdown blocks."""
    try:
        # Case 1: JSON is inside a markdown block
        if "```" in text:
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match: return json.loads(match.group(1))
        
        # Case 2: JSON is embedded in raw text (find first '{' and last '}')
        start, end = text.find('{'), text.rfind('}')
        if start != -1: return json.loads(text[start:end+1])
        
        # Case 3: Text is pure JSON
        return json.loads(text)
    except:
        return {}


# --- NODES (The Agents) ---
def tagger_node(state: ProposalState) -> ProposalState:
    """
    Node 1: The 'Brain' (LLM) - Classify proposal using Groq/Llama3
    Responsibility: Read the proposal and attempt to classify it.

    """
    
    # 1. Safety Check: API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
        
    llm = ChatGroq(model=Config.MODEL_NAME, temperature=Config.TEMPERATURE, api_key=api_key)

    # 2. Optimization: Simplify Taxonomy
    # We strip out complex metadata and send ONLY (Name -> Definition) to the LLM.
    simple_tax = {k: v.get('definition', '') for k, v in state['taxonomy'].items()}

    # 3. Prompt Engineering
    # We explicitly ask for "JSON ONLY" and specific fields (evidence/reasoning).

    prompt = f"""
    Role: Senior Data Classifier.
    Taxonomy: {json.dumps(simple_tax, indent=2)}

    Task:
    1. Classify the proposal using 1-3 tags from the list.
    2. Extract a direct quote (evidence).
    3. Explain reasoning.

    Proposal: "{state['description']}"

    Output JSON ONLY: {{ "tags": [], "evidence": "", "reasoning": "" }}
    """

    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json_robust(res.content)
        state['proposed_tags'] = data.get('tags', [])
        state['evidence'] = data.get('evidence', "")
        state['reasoning'] = data.get('reasoning', "")
    except Exception as e:
        state['reasoning'] = f"LLM Error: {e}"
        state['proposed_tags'] = []

    return state


def validator_node(state: ProposalState) -> ProposalState:
    """
    Node 2: The 'Police' - Deterministic check for hallucinations
    Responsibility: Check for hallucinations and data quality.
    No LLM is used here—pure Python for 100% reliability.
    """

    allowed = set(state['taxonomy'].keys())
    proposed = set(state['proposed_tags'])

    issues = []

    # Check 1: Hallucination (Did the LLM invent a tag?)
    if not proposed.issubset(allowed):
        issues.append(f"Invalid tags: {proposed - allowed}")
    
    # Check 2: Completeness
    if not proposed:
        issues.append("No tags selected")
    
    # Check 3: Evidence Quality (Did it quote enough text?)
    if len(state['evidence']) < Config.MIN_EVIDENCE_CHARS:
        issues.append("Evidence too short")

    state['validation_issues'] = issues
    state['validation_passed'] = len(issues) == 0
    return state


def retry_node(state: ProposalState) -> ProposalState:
    
    """
    Node 3: The 'Loop Manager' - Increment retry counter
    Responsibility: Increment the counter so we don't loop forever.
    """

    state['retag_count'] += 1
    return state


def scorer_node(state: ProposalState) -> ProposalState:
    
    """
    Node 4: The 'Judge' (Scoring) - Calculate confidence score and make final decision
    Responsibility: Calculate final composite score and make PUBLISH/HOLD decision.
    
    Scoring Logic (from README):
    - +0.50: Valid Tags (No Hallucinations)
    - +0.30: Evidence Found (Quote > 30 chars)
    - +0.20: Reasoning Provided (Explanation > 15 chars)
    """

    score = 0.0

    # Calculate Component Scores
    if state['validation_passed']: score += 0.5
    if len(state['evidence']) > 30: score += 0.3
    if len(state['reasoning']) > 15: score += 0.2

    state['confidence_score'] = round(score, 2)

    # Final Threshold Check
    if score >= Config.MIN_CONFIDENCE_TO_PUBLISH:
        state['decision'] = "PUBLISH"
        state['decision_rationale'] = f"Score {score} >= {Config.MIN_CONFIDENCE_TO_PUBLISH}"
    else:
        state['decision'] = "HOLD"
        state['decision_rationale'] = f"Issues: {state['validation_issues']}"

    return state


def should_retry(state: ProposalState) -> Literal["retry", "score"]:

    """
    Conditional Edge Logic:
    Decides whether to loop back to Tagger or move forward to Scoring.
    """
    if not state['validation_passed'] and state['retag_count'] < Config.MAX_RETRY_ATTEMPTS:
        return "retry"
    return "score"


# --- Graph Builder ---
def build_graph():
    """
    Constructs the LangGraph State Machine.
    """
    workflow = StateGraph(ProposalState)

    # 1. Add Nodes
    workflow.add_node("tagger", tagger_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("scorer", scorer_node)

    # 2. Add Edges (The Flow)
    workflow.set_entry_point("tagger")
    workflow.add_edge("tagger", "validator")
    
    # Conditional Routing: Validator -> [Retry OR Scorer]
    workflow.add_conditional_edges(
        "validator",
        should_retry,
        {"retry": "retry", "score": "scorer"}
    )
    
    # Loop: Retry -> Tagger
    workflow.add_edge("retry", "tagger")

    # End: Scorer -> Finish
    workflow.add_edge("scorer", END)

    return workflow.compile()