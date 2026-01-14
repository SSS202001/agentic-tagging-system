import os
import json
import time
import pandas as pd
from src.schemas import Config
from src.graph import build_graph

def main():
    print("🚀 Starting Agentic Pipeline (Modular)...")
    
    # 1. Load Data
    try:
        # Check Inputs
        if not os.path.exists(Config.INPUT_CSV):
            raise FileNotFoundError(f"Missing input CSV at: {Config.INPUT_CSV}")
        
        if not os.path.exists(Config.TAXONOMY_FILE):
            raise FileNotFoundError(
                f"Missing Taxonomy at: {Config.TAXONOMY_FILE}\n"
                "👉 Did you run the 'Taxonomy_Generation.ipynb' notebook first?"
            )
            
        # Read Files
        df = pd.read_csv(Config.INPUT_CSV)
        with open(Config.TAXONOMY_FILE, 'r') as f:
            tax_raw = json.load(f)
            # Handle potential nested 'taxonomy' key from Phase 1 output
            tax_dict = tax_raw.get('taxonomy', tax_raw)
            
        print(f"📂 Loaded {len(df)} proposals and taxonomy.")
        print(f"📂 Output will be saved to: {Config.OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    # 2. Compile Graph
    app = build_graph()
    final_results = []

    # 3. Process Rows
    print(f"⚙️  Processing...")
    for i, row in df.iterrows():
        try:
            # Initialize state with inputs and defaults
            initial_state = {
                "proposal_id": str(row.get('proposalId', i)),
                "description": row['description'],
                "taxonomy": tax_dict,
                "retag_count": 0, "proposed_tags": [], "evidence": "",
                "validation_passed": False, "validation_issues": []
            }
            
            # Run Agent
            final = app.invoke(initial_state)
            
            # Log & Store
            icon = "🟢" if final['decision'] == "PUBLISH" else "🔴"
            print(f"{icon} [{final['proposal_id']}] {final['decision']} (Conf: {final['confidence_score']})")

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
            print(f"❌ Error on row {i}: {e}")

    # 4. Save Results
    res_df = pd.DataFrame(final_results)
    res_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✅ Pipeline Complete. Results saved to: {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()