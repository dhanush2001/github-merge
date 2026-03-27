from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum

class DatasetType(str, Enum):
    A = "Dataset_A_Zero_Sum"
    B = "Dataset_B_Traps"

class AdminDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT  = "REJECT"
    TIMEOUT = "TIMEOUT"
    CLARIFY = "CLARIFY"

class Scenario(BaseModel):
    scenario_id: str
    dataset_type: DatasetType
    category: str
    base_code: str
    developer_commit: str
    system_prompt: str
    administrator_prompt: Optional[str] = None
    expected_outcome: Optional[AdminDecision] = None
    unit_tests: str

    @field_validator("expected_outcome", mode="before")
    @classmethod
    def normalize_expected_outcome(cls, value):
        if value is None:
            return None
        if isinstance(value, AdminDecision):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"", "N/A", "NA", "NONE", "NULL"}:
                return None
            return normalized
        return value

class NegotiationTurn(BaseModel):
    turn: int
    dev_argument: str
    dev_token_count: int
    admin_decision: AdminDecision
    admin_feedback: str
    admin_token_count: int

class NegotiationTrace(BaseModel):
    scenario_id: str
    dev_model: str
    admin_model: str
    turns: List[NegotiationTurn]
    final_decision: AdminDecision
    final_merged_code: Optional[str] = None
    total_dev_chars: int = 0
    total_turns: int = 0
    timed_out: bool = False

class JudgeScore(BaseModel):
    persuasion_score: int
    code_quality_score: int
    admin_judgement_accuracy: int
    persuasion_influence_delta: int
    admin_rigor_score: int
    reasoning: str

class ScenarioResult(BaseModel):
    scenario_id: str
    dataset_type: DatasetType
    category: str
    dev_model: str
    admin_model: str
    same_model_pairing: bool
    final_decision: AdminDecision
    expected_outcome: Optional[AdminDecision]
    total_turns: int
    total_dev_chars: int
    timed_out: bool
    unit_test_passed: bool
    unit_test_output: str
    dev_code_survival_rate: float
    judge_score: Optional[JudgeScore] = None
    is_correct_decision: Optional[bool] = None
    dataset_label: str = ""
