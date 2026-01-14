import os
from typing import TypedDict, List, Literal

class Config:
    """Centralized System Thresholds & Configuration"""

    # --- MODEL SETTINGS ---
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.0
    
    # --- FILE PATH CONFIGURATION ---
    # This robustly finds the 'data' folder relative to this file
    # src/schemas.py -> .. -> root -> data/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Input/Output Artifacts
    INPUT_CSV = os.path.join(DATA_DIR, 'proposals.csv')
    TAXONOMY_FILE = os.path.join(DATA_DIR, 'taxonomy.json')
    OUTPUT_FILE = os.path.join(DATA_DIR, 'tagged_results.csv')

    # --- DECISION LOGIC THRESHOLDS ---
    # These control the "Publish vs Hold" gate.
    MIN_CONFIDENCE_TO_PUBLISH = 0.65
    MAX_RETRY_ATTEMPTS = 2
    MIN_EVIDENCE_CHARS = 10


class ProposalState(TypedDict):
    """
    The Memory Object passed between graph nodes.
    """
    # --- 1. Static Inputs (Read-Only) ---
    proposal_id: str
    description: str
    taxonomy: dict

    # --- 2. Model Outputs (Mutable) ---
    # The Tagger node writes these.
    proposed_tags: List[str]
    evidence: str
    reasoning: str

    # --- 3. Internal Logic Flags (Control Flow) ---
    # The Validator and Retry nodes use these to direct traffic.
    validation_passed: bool
    validation_issues: List[str]
    retag_count: int

    # --- 4. Final Outputs (Reporting) ---
    # The Scorer node finalizes these for the CSV export
    confidence_score: float
    decision: Literal["PUBLISH", "HOLD"]
    decision_rationale: str