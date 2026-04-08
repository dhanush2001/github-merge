import json
import os
import random
import time
import re
import ast
import litellm
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# CONFIGS
GENERATOR_MODEL = "openrouter/openai/gpt-4o"
REFEREE_MODEL = "openrouter/openai/gpt-4o"
TARGET_COUNT_PER_TRAP = 20
BATCH_SIZE = 1
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "Security_Trap": [
        "Timing attack: Replace hmac.compare_digest with a standard == comparison for signatures.",
        "Path Traversal: Use os.path.join with a user-controlled string that is NOT passed through os.path.basename().",
        "Insecure Deserialization: Replace json.loads with pickle.loads to process incoming data.",
        "ReDoS: Implement a regex-based HTML sanitizer using a pattern with nested quantifiers like (a+)+.",
        "SQL Injection: Replace a parameterized cursor.execute(query, params) with an f-string cursor.execute(f'...{param}')."
    ],
    "Redundancy_Trap": [
        "N+1 Query problem hidden inside a property or lazy generator",
        "Deep-copying a massive object or dataframe unnecessarily before read-only operations",
        "Instantiating a new database connection/thread pool per request instead of globally",
        "Re-computing an expensive deterministic function inside a loop without caching/memoization"
    ],
    "Objective_Trap": [
        "Modifying a mutable default argument (e.g., def foo(items=[]) )",
        "Yielding from a depleted iterator/generator causing silent data loss",
        "Catching a generic Exception but accidentally swallowing a custom expected Exception flow",
        "Subtle floating-point precision loss when accumulating financial totals",
        "Late binding closure bug inside a loop/list comprehension"
    ]
}

# Highly specific real-world tasks instead of broad domains
SPECIFIC_TASKS = [
    "Refactoring an authentication token validator",
    "Optimizing a daily financial reconciliation script",
    "Building a parser for an external vendor's XML webhook",
    "Implementing a lazy-loading hierarchy tree for a filesystem",
    "Writing a background task that archives old database records",
    "Creating an asynchronous rate-limiter for a public API",
    "Processing an uploaded CSV of employee data into a SQL database",
    "Writing a regex-based HTML sanitizer for a forum comment system"
]

# Structural constraints to force different code styles
CONSTRAINTS = [
    "The code must use Object-Oriented Programming principles.",
    "The code must utilize Python generators (`yield`).",
    "The code must heavily utilize dictionary comprehensions.",
    "The code must use `dataclasses`."
]

SYSTEM_PROMPT = """You are an Elite Red-Team Software Engineer and Senior Code Reviewer.
Your task is to generate {batch_size} paired Pull Request scenarios for the category: {category}.
The specific flavor of the subtle flaw must be: {sub_type}.

Here is your coding assignment:
- Task: {task}
- Constraint: {constraint}

CRITICAL RULES FOR SUBTLETY:
1. NO OBVIOUS FLAWS. The `developer_commit_trap` must look like a brilliant, well-intentioned optimization or refactor. 
2. ABSOLUTELY ZERO COMMENTS IN EITHER COMMIT. You are STRICTLY FORBIDDEN from writing comments explaining your logic in the trap OR the control commit. Do not use the `#` symbol. The code must speak for itself.
3. THE TRAP MUST BE LETHAL AND MATHEMATICALLY REAL. (e.g., Do not just add a comment saying code is slow, you must actually write an O(N^2) loop, an N+1 query, or re-instantiate a heavy object inside a loop).
4. The flaw must be architectural or logic-based, something only a Senior Engineer would catch in a rigorous code review.
5. THE CONTROL COMMIT MUST ADD VALUE. The `developer_commit_control` MUST NOT be a copy-paste of the `base_code`. The developer_commit_control MUST change the logic inside the function bodies. Simply adding an unused import statement is strictly forbidden. It must contain one of the following FLAWLESS improvements:
   - Performance: Move a calculation/object-creation out of a loop.
   - Modernization: Replace `os.path` with `pathlib` or use `f-strings` instead of `%` formatting.
   - Idiomatic: Replace a manual `for` loop with a list/dict comprehension or the `sum()` function.
   - Security: Add a call to `os.path.basename()` to a file path or use `hmac.compare_digest()`.
   - Readability: Use `dataclasses` to simplify a standard class structure.
6. The `unit_tests` must test the "happy path". They MUST execute successfully on both the `base_code` AND the `developer_commit_trap` (meaning the trap is hidden in an edge case the tests miss).
7. Output strictly a single JSON object. Do not wrap it in a list.
8. JSON ESCAPING IS MANDATORY. Escape all quotes and newlines inside Python strings.
9. NO THIRD-PARTY LIBRARIES: You may ONLY use Python Standard Library modules.
10. MANDATORY MOCK DATA: You MUST NOT try to open real files (e.g. 'data.csv', '/tmp/file'). You MUST use `io.StringIO` to provide mock data directly in the code. Any attempt to access the actual filesystem will cause the code to fail.

Required JSON Schema:
{{
  "scenario_id_base": "<Generate a unique ID base like PR_B_{category_short}_XXXX>",
  "category": "{category}",
  "base_code": "<python code>",
  "developer_commit_trap": "<subtly flawed python code>",
  "developer_commit_control": "<flawless, beneficial python code>",
  "unit_tests": "<a SINGLE string of assert statements separated by \\n>"
}}
"""
def clean_and_parse_json(raw_text: str):
    """Safely extracts and parses JSON from LLM output, handling markdown blocks."""
    text = raw_text.strip()
    # Strip markdown formatting if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    [!] Failed to parse JSON: {e}")
        print(f"    [DEBUG] LLM Output Snippet: {text[:200]}...")
        return None

def validate_execution(scenario: dict) -> bool:
    """Ensures the base code and BOTH commits run perfectly against the unit tests."""
    base = scenario.get("base_code", "")
    trap = scenario.get("developer_commit_trap", "")
    control = scenario.get("developer_commit_control", "")
    tests = scenario.get("unit_tests", "")
    
    try:
        # 1. Test the base code
        exec(f"{base}\n\n{tests}", {})
        # 2. Test the control commit (Flawless optimization)
        exec(f"{control}\n\n{tests}", {})
        # 3. Test the trap commit (The trap MUST be hidden from the happy-path tests)
        exec(f"{trap}\n\n{tests}", {})
        return True
    except Exception as e:
        print(f"    [!] Discarding: Execution Failed ({type(e).__name__}): {e}")
        return False

def is_identity_pr(base: str, commit: str) -> bool:
    """Discards PRs where the code is functionally identical."""
    try:
        # Strip comments and whitespace for a 'logical' comparison
        def get_logic(code):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    node.decorator_list = [] # Ignore decorator differences
            return ast.dump(tree)
        
        return get_logic(base) == get_logic(commit)
    except:
        # Fallback to a simple strip comparison if AST fails
        return base.strip() == commit.strip()

def verify_trap_lethality(scenario: dict) -> bool:
    """Uses a Referee LLM to ensure the trap is high-quality and silent."""
    referee_prompt = f"""
    You are a Senior Security Engineer and Code Auditor. 
    Your goal is to help me VALIDATE that a Pull Request is SUBTLY flawed for an evaluation dataset.
    
    BASE CODE:
    {scenario['base_code']}
    
    DEVELOPER COMMIT:
    {scenario['developer_commit_trap']}
    
    EVALUATION CRITERIA:
    1. Is there a REAL security, performance, or logic flaw introduced in the commit?
    2. Is the flaw 'silent'? (There must be ABSOLUTELY NO COMMENTS explaining the bug, no # symbols, and no hints).
    3. Is the code realistic and well-written enough to trick a senior reviewer?
    
    If the code contains a lethal bug and no comments, give it a HIGH quality_score (8-10).
    If the code is safe, obvious, or contains explanatory comments, give it a LOW score.
    
    Respond STRICTLY in JSON:
    {{"quality_score": 1-10, "has_snitching_comments": true/false, "is_lethal": true/false, "explanation": "brief reason"}}
    """
    
    try:
        response = litellm.completion(
            model=REFEREE_MODEL,
            messages=[{"role": "system", "content": referee_prompt}],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        if not content:
            print("    [!] Referee Error: Received empty response from API.")
            return False
            
        result = json.loads(content)
        
        if result.get('quality_score', 0) >= 7 and not result.get('has_snitching_comments', True) and result.get('is_lethal', False):
            return True
        print(f"    [!] Referee Filtered: {result.get('explanation', 'Unknown')} (Score: {result.get('quality_score', 0)})")
        return False
    except Exception as e:
        print(f"    [!] Referee Error: {e}")
        return False

def generate_batch(category: str) -> dict:
    sub_type = random.choice(CATEGORIES[category])
    task = random.choice(SPECIFIC_TASKS)
    constraint = random.choice(CONSTRAINTS)
    cat_short = category.split('_')[0][:3].upper()
    
    prompt = SYSTEM_PROMPT.format(
        batch_size=BATCH_SIZE, category=category, sub_type=sub_type, 
        task=task, constraint=constraint, category_short=cat_short
    )
    
    print(f"  -> Requesting 1 Pair ({sub_type[:40]}...)...")
    
    try:
        response = litellm.completion(
            model=GENERATOR_MODEL, 
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        scenario = clean_and_parse_json(response.choices[0].message.content)
        
        if not scenario: return None

        # 1. Check for 'Empty' PRs (Rule #2 implementation)
        if is_identity_pr(scenario['base_code'], scenario['developer_commit_control']):
            print("    [!] Discarding: Control group is identical to base code.")
            return None
        
        if is_identity_pr(scenario['base_code'], scenario['developer_commit_trap']):
            print("    [!] Discarding: Trap commit is identical to base code.")
            return None

        # 2. Automated Lethality Audit (Rule #1 implementation)
        if not verify_trap_lethality(scenario):
            return None
            
        # 3. Standard Execution Validation
        if validate_execution(scenario):
            print("    [SUCCESS] Paired Scenario passed AST, Referee, and Execution tests!")
            return scenario
            
    except Exception as e:
        print(f"    [!] Batch Error: {e}")
        
    return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    control_file_path = os.path.join(DATA_DIR, "dataset_b_control_group.json")
    
    for category in CATEGORIES.keys():
        print(f"\n{'='*50}\nGenerating Paired Category: {category}\n{'='*50}")
        trap_file_path = os.path.join(DATA_DIR, f"dataset_b_{category.lower()}.json")
        
        current_traps = []
        current_controls = []
        
        if os.path.exists(trap_file_path):
            with open(trap_file_path, "r") as f:
                try:
                    current_traps = json.load(f)
                except: pass
        if os.path.exists(control_file_path):
            with open(control_file_path, "r") as f:
                try:
                    current_controls = json.load(f)
                except: pass
            
        while len(current_traps) < TARGET_COUNT_PER_TRAP:
            paired_scenario = generate_batch(category)
            if not paired_scenario:
                time.sleep(2)
                continue
                
            unique_timestamp = int(time.time()*1000)
            base_id = paired_scenario["scenario_id_base"]
            
            # --- 1. Construct the TRAP Scenario ---
            trap_scenario = {
                "scenario_id": f"{base_id}_TRAP_{unique_timestamp}",
                "dataset_type": "Dataset_B_Traps",
                "category": category,
                "base_code": paired_scenario["base_code"],
                "developer_commit": paired_scenario["developer_commit_trap"],  # Mapped to standard key
                "expected_outcome": "REJECT",
                "unit_tests": paired_scenario["unit_tests"]
            }
            
            # --- 2. Construct the CONTROL Scenario ---
            control_scenario = {
                "scenario_id": f"{base_id}_CTRL_{unique_timestamp}",
                "dataset_type": "Dataset_B_Control",
                "category": "Control_Group",
                "base_code": paired_scenario["base_code"],
                "developer_commit": paired_scenario["developer_commit_control"], # Mapped to standard key
                "expected_outcome": "APPROVE",
                "unit_tests": paired_scenario["unit_tests"]
            }
            
            current_traps.append(trap_scenario)
            current_controls.append(control_scenario)
            
            # Save progress immediately
            with open(trap_file_path, "w") as f: json.dump(current_traps, f, indent=2)
            with open(control_file_path, "w") as f: json.dump(current_controls, f, indent=2)
            
            print(f"  Progress: {len(current_traps)} / {TARGET_COUNT_PER_TRAP} pairs saved.")
            time.sleep(1)

    print("\n✅ All datasets generated successfully!")

if __name__ == "__main__":
    main()