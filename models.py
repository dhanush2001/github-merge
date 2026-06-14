from pydantic import BaseModel, field_validator
from typing import Optional, List
from enum import Enum

class DatasetType(str, Enum):
    B = "Dataset_B_Traps"
    B_CONTROL = "Dataset_B_Control"

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
    expected_outcome: Optional[AdminDecision] = None
    unit_tests: str

    @field_validator("dataset_type", mode="before")
    @classmethod
    def normalize_dataset_type(cls, value):
        if isinstance(value, DatasetType):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            aliases = {
                "Dataset_B_Traps_V2": DatasetType.B,
                "Dataset_B_Control_V2": DatasetType.B_CONTROL,
            }
            if normalized in aliases:
                return aliases[normalized]
            return normalized
        return value

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
    dev_char_count: int
    dev_input_token_count: int = 0
    dev_token_count: int = 0
    admin_decision: AdminDecision
    admin_feedback: str
    admin_char_count: int
    admin_input_token_count: int = 0
    admin_token_count: int = 0

class NegotiationTrace(BaseModel):
    scenario_id: str
    dev_model: str
    admin_model: str
    turns: List[NegotiationTurn]
    final_decision: AdminDecision
    final_merged_code: Optional[str] = None
    total_dev_chars: int = 0
    total_dev_input_tokens: int = 0
    total_dev_tokens: int = 0
    total_admin_chars: int = 0
    total_admin_input_tokens: int = 0
    total_admin_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
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
    total_dev_input_tokens: int = 0
    total_dev_tokens: int = 0
    total_admin_chars: int = 0
    total_admin_input_tokens: int = 0
    total_admin_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    timed_out: bool
    unit_test_passed: bool
    unit_test_output: str
    dev_code_survival_rate: float
    judge_score: Optional[JudgeScore] = None
    is_correct_decision: Optional[bool] = None
    dataset_label: str = ""
    persuasion_mode: str = "full"
    turns: List[NegotiationTurn] = []
