import os
import json
import time
import pandas as pd
from src.schemas import Config
from src.graph import build_graph

def main():
    print("🚀 Starting Agentic Pipeline (Modular)...")
    
    # --- STEP 1: LOAD & VALIDATE INPUTS ---
    try:
        # Sanity check: Ensure our input files actually exist before we start
        if not os.path.exists(Config.INPUT_CSV):
            raise FileNotFoundError(f"Missing input CSV at: {Config.INPUT_CSV}")
        
        if not os.path.exists(Config.TAXONOMY_FILE):
            raise FileNotFoundError(
                f"Missing Taxonomy at: {Config.TAXONOMY_FILE}\n"
                "👉 Did you run the 'Taxonomy_Generation.ipynb' notebook first?"
            )
            
        # Load the raw data
        df = pd.read_csv(Config.INPUT_CSV)

        # Load the taxonomy schema
        # We handle two cases here: 
        # 1. The JSON is a direct dictionary of categories
        # 2. The JSON is wrapped in a "taxonomy" key (common artifact from Phase 1)
        with open(Config.TAXONOMY_FILE, 'r') as f:
            tax_raw = json.load(f)
            # Handle potential nested 'taxonomy' key from Phase 1 output
            tax_dict = tax_raw.get('taxonomy', tax_raw)
            
        print(f"📂 Loaded {len(df)} proposals and taxonomy.")
        print(f"📂 Output will be saved to: {Config.OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    # --- STEP 2: BUILD THE AGENT ---
    # Compile the LangGraph state machine once. 
    app = build_graph()
    final_results = []

    # --- STEP 3: RUN THE PIPELINE ---
    print(f"⚙️  Processing...")
    for i, row in df.iterrows():
        try:
            # Create a fresh state object for this specific proposal.
            initial_state = {
                "proposal_id": str(row.get('proposalId', i)),
                "description": row['description'],
                "taxonomy": tax_dict,
                "retag_count": 0, "proposed_tags": [], "evidence": "",
                "validation_passed": False, "validation_issues": []
            }
            
            # Run the Agent
            final = app.invoke(initial_state)
            
            # Visual feedback for the console user
            icon = "🟢" if final['decision'] == "PUBLISH" else "🔴"
            print(f"{icon} [{final['proposal_id']}] {final['decision']} (Conf: {final['confidence_score']})")

            # Collect the structured output
            final_results.append({
                "id": final['proposal_id'],
                "description": row['description'],
                "tags": ", ".join(final['proposed_tags']),
                "decision": final['decision'],
                "confidence": final['confidence_score'],
                "evidence": final['evidence'],
                "rationale": final['decision_rationale']
            })
            
            time.sleep(0.5) # Rate limit politeness

        except Exception as e:
            print(f"❌ Error on row {i}: {e}")   # Graceful failure: If one row crashes, don't kill the whole script.

    # --- STEP 4: EXPORT THE RESULTS ---
    res_df = pd.DataFrame(final_results)
    res_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✅ Pipeline Complete. Results saved to: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()