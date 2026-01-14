import os
from typing import TypedDict, List, Literal

class Config:
    """Centralized System Thresholds & Configuration"""
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.0
    
    # --- FILE PATH CONFIGURATION ---
    # This robustly finds the 'data' folder relative to this file
    # src/schemas.py -> .. -> root -> data/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    INPUT_CSV = os.path.join(DATA_DIR, 'proposals.csv')
    TAXONOMY_FILE = os.path.join(DATA_DIR, 'taxonomy.json')
    OUTPUT_FILE = os.path.join(DATA_DIR, 'tagged_results.csv')

    # Logic Thresholds
    MIN_CONFIDENCE_TO_PUBLISH = 0.65
    MAX_RETRY_ATTEMPTS = 2
    MIN_EVIDENCE_CHARS = 10

class ProposalState(TypedDict):
    """
    The Memory Object passed between graph nodes.
    """
    # --- Inputs ---
    proposal_id: str
    description: str
    taxonomy: dict

    # --- LLM Outputs ---
    proposed_tags: List[str]
    evidence: str
    reasoning: str

    # --- Internal Logic ---
    validation_passed: bool
    validation_issues: List[str]
    retag_count: int

    # --- Final Decision ---
    confidence_score: float
    decision: Literal["PUBLISH", "HOLD"]
    decision_rationale: str