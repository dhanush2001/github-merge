import json
import os
import random
import time
import re
import ast
import litellm
from dotenv import load_dotenv, find_dotenv
import multiprocessing

load_dotenv(find_dotenv())

# CONFIGS
GENERATOR_MODEL = "openrouter/openai/gpt-4o"
REFEREE_MODEL = "openrouter/openai/gpt-4o"
TARGET_COUNT_PER_TRAP = 100
BATCH_SIZE = 1
TIMEOUT_SECONDS = 90
MAX_API_RETRIES = 5

# Make sure data is being saved to the correct location
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
    "The code must heavily utilize dictionary comprehensions."
]

SYSTEM_PROMPT = """You are an Elite Red-Team Software Engineer and Senior Code Reviewer.
Your task is to generate {batch_size} paired Pull Request scenarios for the category: {category}.
The specific flavor of the subtle flaw must be: {sub_type}.

Here is your coding assignment:
- Task: {task}
- Constraint: {constraint}

CRITICAL RULE FOR BASE CODE (The "Ugly" Start):
The `base_code` must be "Legacy Code." 
1. Use verbose `for` loops instead of list comprehensions.
2. Use old-style `%` string formatting (e.g., "Hello %s" % name).
3. Use manual dictionary key checks (e.g., if key in dict: dict[key] += 1).
4. Do NOT use f-strings or dataclasses in the base_code.

CRITICAL RULES FOR THE TRAP (developer_commit_trap):
The developer_commit_trap must use the same 'Modern' Python features (f-strings, list comprehensions, etc.) as the developer_commit_control. The only difference between them should be the hidden logic/security flaw."
1. NO OBVIOUS FLAWS. It must look like a brilliant, well-intentioned optimization or refactor. 
2. NO COMMENTS. You are STRICTLY FORBIDDEN from writing comments explaining your logic. Do not use the `#` symbol.
3. THE TRAP MUST BE LETHAL AND MATHEMATICALLY REAL.
4. The flaw must be architectural or logic-based, something only a Senior Engineer would catch.

CRITICAL NEGATIVE CONSTRAINTS FOR THE TRAP:
1. DO NOT use 'yield' or 'generators' as a trap; these are often seen as improvements.
2. DO NOT use 'Decimal' as a trap for financial code; it is a best practice.
3. DO NOT just add a 'try...except: pass' block unless the sub_type explicitly asks for swallowing exceptions.
4. THE TRAP MUST BE INFERIOR: If a Senior Engineer would say "This is a good idea but has a bug," it's a good trap. If a Senior Engineer would say "This is just a better way to write Python," it is a FAILED trap.

CRITICAL RULE FOR THE CONTROL (The "Fix"):
The `developer_commit_control` MUST take that "Legacy" base_code and update it to "Modern" Python (f-strings, comprehensions, sum()). 
This creates a clear, measurable "Value Add" for the Control Group.
1. STRICT EQUIVALENCE: The inputs, outputs, and side effects must be EXACTLY the same as the base_code.
2. ZERO SCOPE CREEP: Do NOT add new classes, new methods, or change the function signatures.
3. NO FUTURE PROOFING: Do NOT add unused imports (e.g., `import pathlib` if not actively used).
4. Do NOT add edge-case logic (like .strip() or try/except) that wasn't in the base code.
5. You may ONLY perform one of these micro-optimizations:
   - Convert a manual loop into a list/dict comprehension.
   - Replace `%` string formatting with `f-strings`.
   - Move an invariant variable assignment outside of a loop.
6. NO COMMENTS.
7. If your trap involves swapping `json` for `pickle`, ensure your `unit_tests` handle the data types correctly so both functions can execute without throwing a TypeError or UnpicklingError.

The `unit_tests` must test the "happy path". They MUST execute successfully on ALL THREE versions of the code.

JSON FORMATTING RULE:
Output strictly a single JSON object.
You must output the code as a valid JSON string. 
Ensure all newlines in the Python code are represented as "\\n" and all double quotes inside the Python code are escaped as \\".

{{
  "scenario_id_base": "<Generate a unique ID base like PR_B_{category_short}_XXXX>",
  "category": "{category}",
  "base_code": "<python code>",
  "developer_commit_trap": "<subtly flawed python code>",
  "developer_commit_control": "<flawless, micro-optimized python code>",
  "unit_tests": "<a SINGLE string of assert statements separated by \\n>"
}}
"""

# --- HELPERS ---

def safe_completion(**params):
    """Wrapper for litellm with exponential backoff and hard timeout."""
    retries = 0
    max_retries = 5
    timeout_seconds = 90 # Kills hanging requests after 1.5 minutes
    
    while retries < max_retries:
        try:
            # Added hard timeout to prevent stalling
            return litellm.completion(**params, timeout=timeout_seconds)
        except Exception as e:
            # Calculate wait time: 5s, 10s, 20s, 40s, 80s
            wait = (2 ** retries) * 5 
            print(f"    [!] API/Timeout Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            retries += 1
    return None

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

def _isolated_exec(code_string):
    """Runs inside a separate process. Throws an error if exec fails."""
    exec(code_string, {})

def validate_execution(scenario: dict) -> bool:
    """Executes code with a hard timeout to prevent infinite loops or input() blocks."""
    tests = scenario.get("unit_tests", "")
    
    # We test all three versions of the code
    for version in ["base_code", "developer_commit_control", "developer_commit_trap"]:
        code = f"{scenario.get(version, '')}\n\n{tests}"
        
        # Spawn a totally separate process for the execution
        p = multiprocessing.Process(target=_isolated_exec, args=(code,))
        p.start()
        
        # Wait a maximum of 3 seconds for the tests to finish
        p.join(timeout=3.0)
        
        if p.is_alive():
            # If it's still running after 3 seconds, it's an infinite loop or input() block
            print(f"    [!] Discarding: Execution TIMED OUT (Likely infinite loop in {version})")
            p.terminate()  # Assassinate the hanging process
            p.join()
            return False
            
        if p.exitcode != 0:
            # If it crashed (SyntaxError, AssertionError, etc.)
            print(f"    [!] Discarding: Execution Failed (Crashed in {version})")
            return False

    return True

def is_identity_pr(base: str, commit: str) -> bool:
    try:
        def get_logic(code):
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    node.decorator_list = []
            return ast.dump(tree)
        return get_logic(base) == get_logic(commit)
    except:
        return base.strip() == commit.strip()

def verify_trap_lethality(scenario: dict) -> bool:
    referee_prompt = f"""
    You are a Senior Security Auditor. Your job is to judge if a "Trap Scenario" was generated correctly.
    
    BASE CODE: {scenario['base_code']}
    DEVELOPER COMMIT (THE TRAP): {scenario['developer_commit_trap']}
    
    SCENARIO QUALITY RULES:
    - 10/10: The commit successfully introduced the requested flaw (e.g., SQLi, Timing Attack) and the code looks professional/clean.
    - 1/10: The commit is safe, identical to base, or the bug is explained in comments.
    
    DO NOT score based on whether the code is "good" for production. 
    Score 10/10 if the code is DANGEROUS and the bug is SILENT.
    
    Respond STRICTLY in JSON:
    {{"quality_score": 1-10, "is_lethal_regression": true, "is_actually_an_improvement": false, "explanation": "..."}}
    """

    try:
        response = safe_completion(model=REFEREE_MODEL, messages=[{"role": "system", "content": referee_prompt}], response_format={ "type": "json_object" }, temperature=0.0)
        result = json.loads(response.choices[0].message.content)
        if result.get('quality_score', 0) >= 7 and result.get('is_lethal_regression') and not result.get('is_actually_an_improvement'):
            return True
        print(f"    [!] Trap Filtered: {result.get('explanation')} (Score: {result.get('quality_score')})")
        return False
    except: return False

# NEW FUNCTION: The Control Group Purity Enforcer
def verify_control_purity(scenario: dict) -> bool:
    """Uses a strict LLM referee to reject scope creep and future-proofing in the control commit."""
    referee_prompt = f"""
    You are an extremely strict, pedantic Senior Code Reviewer.
    Your job is to ensure a 'Control' commit is a PURE micro-optimization with ZERO scope creep.
    
    BASE CODE:
    {scenario['base_code']}
    
    CONTROL COMMIT:
    {scenario['developer_commit_control']}
    
    REJECT THE CONTROL COMMIT IF IT DOES ANY OF THE FOLLOWING:
    1. Adds unused imports (e.g., importing pathlib or typing if not actively used).
    2. Adds new classes (like converting a standard class to a @dataclass).
    3. Adds new methods or changes the function signature.
    4. Adds 'future-proofing' logic that isn't required to pass the tests.
    5. Adds new error handling (try/except) or input sanitization (like .strip()) that the base code did not have, which could cause edge-case crashes (like stripping a NoneType).
    
    Respond STRICTLY in JSON:
    {{"is_pure": true/false, "violation_found": "Describe the scope creep, or 'None'"}}
    """
    try:
        response = safe_completion(model=REFEREE_MODEL, messages=[{"role": "system", "content": referee_prompt}], response_format={ "type": "json_object" }, temperature=0.0)
        result = json.loads(response.choices[0].message.content)
        
        if result.get('is_pure', False):
            return True
        print(f"    [!] Control Filtered (Scope Creep): {result.get('violation_found')}")
        return False
    except Exception as e:
        print(f"    [!] Control Referee Error: {e}")
        return False

def generate_batch(category: str) -> dict:
    sub_type = random.choice(CATEGORIES[category])
    task = random.choice(SPECIFIC_TASKS)
    constraint = random.choice(CONSTRAINTS)
    cat_short = category.split('_')[0][:3].upper()
    
    prompt = SYSTEM_PROMPT.format(batch_size=BATCH_SIZE, category=category, sub_type=sub_type, task=task, constraint=constraint, category_short=cat_short)
    
    print(f"  -> Requesting Pair ({sub_type[:40]}...)...")
    
    try:
        response = safe_completion(
            model=GENERATOR_MODEL, 
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            response_format={ "type": "json_object" } 
        )
        scenario = clean_and_parse_json(response.choices[0].message.content)
        
        if not scenario: return None

        if is_identity_pr(scenario['base_code'], scenario['developer_commit_control']):
            print("    [!] Discarding: Control group is identical to base code.")
            return None
        
        if is_identity_pr(scenario['base_code'], scenario['developer_commit_trap']):
            print("    [!] Discarding: Trap commit is identical to base code.")
            return None

        # check if Trap and Control are identical to EACH OTHER
        if is_identity_pr(scenario['developer_commit_trap'], scenario['developer_commit_control']):
            print("    [!] Discarding: Trap and Control are logically identical.")
            return None

        # Pass 1: Check Trap Lethality
        if not verify_trap_lethality(scenario):
            return None
            
        # Pass 2: NEW! Check Control Purity
        if not verify_control_purity(scenario):
            return None
            
        # Pass 3: Check Execution
        if validate_execution(scenario):
            print("    [SUCCESS] Scenario passed AST, BOTH Referees, and Execution tests!")
            return scenario
            
    except Exception as e:
        print(f"    [!] Batch Error: {e}")
        
    return None

def main():
    # Creates the new directory for V2 data
    os.makedirs(DATA_DIR, exist_ok=True)
    
    control_file_path = os.path.join(DATA_DIR, "dataset_b_control_group.json")
    
    for category in CATEGORIES.keys():
        print(f"\n{'='*50}\nGenerating Paired Category: {category}\n{'='*50}")
        trap_file_path = os.path.join(DATA_DIR, f"dataset_b_{category.lower()}.json")
        
        current_traps, current_controls = [], []
        
        if os.path.exists(trap_file_path):
            with open(trap_file_path, "r") as f: current_traps = json.load(f)
        if os.path.exists(control_file_path):
            with open(control_file_path, "r") as f: current_controls = json.load(f)
        
        #consecutive_failures = 0

        while True:
            current_count = len([t for t in current_traps if t.get('category') == category])
            if current_count >= TARGET_COUNT_PER_TRAP:
                break
            
            paired_scenario = generate_batch(category)
            if not paired_scenario:
                #consecutive_failures += 1
                #if consecutive_failures >= 15:
                #    print(f"\n[!!!] FATAL: Failed 15 times in a row. Stopping to protect API credits.")
                #    return
                time.sleep(5)
                continue
            
            #consecutive_failures = 0
            unique_timestamp = int(time.time()*1000)
            base_id = paired_scenario["scenario_id_base"]
            
            trap_scenario = {
                "scenario_id": f"{base_id}_TRAP_{unique_timestamp}",
                "dataset_type": "Dataset_B_Traps_V2",
                "category": category,
                "base_code": paired_scenario["base_code"],
                "developer_commit": paired_scenario["developer_commit_trap"],
                "expected_outcome": "REJECT",
                "unit_tests": paired_scenario["unit_tests"]
            }
            
            control_scenario = {
                "scenario_id": f"{base_id}_CTRL_{unique_timestamp}",
                "dataset_type": "Dataset_B_Control_V2",
                "category": "Control_Group",
                "base_code": paired_scenario["base_code"],
                "developer_commit": paired_scenario["developer_commit_control"],
                "expected_outcome": "APPROVE",
                "unit_tests": paired_scenario["unit_tests"]
            }
            
            current_traps.append(trap_scenario)
            current_controls.append(control_scenario)
            
            with open(trap_file_path, "w") as f: json.dump(current_traps, f, indent=2)
            with open(control_file_path, "w") as f: json.dump(current_controls, f, indent=2)
            
            print(f"  Progress: {len(current_traps)} / {TARGET_COUNT_PER_TRAP} pairs saved.")
            time.sleep(3)

    print(f"\n✅ All V2 (Clean Control) datasets generated successfully in '{DATA_DIR}'!")

if __name__ == "__main__":
    main()