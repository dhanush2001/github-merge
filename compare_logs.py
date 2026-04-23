import json
import os

# FILE NAMES
FILE_PERSUASION = "results/April_8th_60Items_Persuasion_Results/conversation_logs_20260408_200530.json"
FILE_NO_PERSUASION = "results/April_12th_60Items_No_Persuasion_Results/conversation_logs_20260412_151010.json"

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return []
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    print(f"Loading Persuasion logs from: {FILE_PERSUASION}")
    print(f"Loading No Persuasion logs from:    {FILE_NO_PERSUASION}\n")
    
    persuasion_data = load_json(FILE_PERSUASION)
    no_persuasion_data = load_json(FILE_NO_PERSUASION)
    
    if not persuasion_data or not no_persuasion_data:
        return

    # Index by scenario_id so we can easily compare them side-by-side
    persuasion_dict = {item["scenario_id"]: item for item in persuasion_data}
    no_persuasion_dict = {item["scenario_id"]: item for item in no_persuasion_data}

    # Statistics counters
    stats = {
        "total_compared": 0,
        "both_approved": 0,
        "both_rejected": 0,
        "persuasion_won": 0,  # Persuasion got Approve, No Persuasion got Reject
        "no_persuasion_won": 0,     # No Persuasion got Approve, Persuasion got Reject
    }
    
    # Keep track of the specific scenarios to print later
    mismatches = []

    for scenario_id, p_item in persuasion_dict.items():
        if scenario_id not in no_persuasion_dict:
            continue
            
        stats["total_compared"] += 1
        
        c_item = no_persuasion_dict[scenario_id]
        
        # Check if the decision contains APPROVE
        p_approved = "APPROVE" in str(p_item.get("final_decision", ""))
        c_approved = "APPROVE" in str(c_item.get("final_decision", ""))
        category = p_item.get("category", "Unknown")

        if p_approved and c_approved:
            stats["both_approved"] += 1
        elif not p_approved and not c_approved:
            stats["both_rejected"] += 1
        elif p_approved and not c_approved:
            stats["persuasion_won"] += 1
            mismatches.append(f"  [Persuasion WON] {category}: {scenario_id}")
        elif not p_approved and c_approved:
            stats["no_persuasion_won"] += 1
            mismatches.append(f"  [No Persuasion WON]    {category}: {scenario_id}")

    # Print Report
    print("="*50)
    print("                 FINAL STATISTICS")
    print("="*50)
    print(f"Total Scenarios Compared: {stats['total_compared']}")
    print("-" * 50)
    print(f"AGREEMENTS: {stats['both_approved'] + stats['both_rejected']}")
    print(f"  - Both Approved: {stats['both_approved']}")
    print(f"  - Both Rejected: {stats['both_rejected']}")
    print("-" * 50)
    print(f"DISAGREEMENTS: {stats['persuasion_won'] + stats['no_persuasion_won']}")
    print(f"  - Persuasion got APPROVE / No Persuasion got REJECT: {stats['persuasion_won']}")
    print(f"  - No Persuasion got APPROVE / Persuasion got REJECT: {stats['no_persuasion_won']}")
    print("="*50)
    
    if mismatches:
        print("\nBreakdown of Disagreements (Scenario IDs):")
        for m in sorted(mismatches):
            print(m)

if __name__ == "__main__":
    main()