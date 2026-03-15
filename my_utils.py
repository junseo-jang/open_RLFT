"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import ast
import asyncio
import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from typing import Any
import re

from open_instruct import logger_utils
from open_instruct.IFEvalG import instructions_registry
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

logger = logger_utils.setup_logger(__name__)

# remove excessive logging from liteLLM
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("litellm.cost_calculator").setLevel(logging.CRITICAL)
logging.getLogger("litellm._client").setLevel(logging.CRITICAL)
logging.getLogger("cost_calculator").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


@dataclasses.dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None


@dataclasses.dataclass
class VerifierConfig:
    """For now this config exists to support LMJudgeVerifer, can be expanded to support other verifers"""

    @classmethod
    def from_args(cls, *arg_sources) -> "VerifierConfig":
        """
        Create a VerifierConfig from multiple argument sources by automatically matching field names.
        Only fields that exist in both the sources and VerifierConfig will be passed through.
        Later sources override earlier ones if they have the same field.
        """
        verifier_fields = {f.name for f in dataclasses.fields(cls)}

        matching_kwargs = {}
        for source in arg_sources:
            if source is None:
                continue
            for field_name in verifier_fields:
                if hasattr(source, field_name):
                    matching_kwargs[field_name] = getattr(source, field_name)

        return cls(**matching_kwargs)


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: VerifierConfig | None = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return VerifierConfig

    @abstractmethod
    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query
            rollout_state (Optional[dict]): Rollout state dict (rewards, step_count, done) for env verifiers.

        Returns:
            VerificationResult
        """

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.
            rollout_state (Optional[dict]): Rollout state dict for env verifiers.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.__call__(tokenized_prediction, prediction, label, query, rollout_state)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint(s) are a list of constraint ids.
    This list is found under the key "instruction_id" in the ground_truth dict.

    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | dict,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        instruction_dict = instructions_registry.INSTRUCTION_DICT
        constraint_dict = ast.literal_eval(label)
        constraint_dict = constraint_dict[0]
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
        answer = remove_thinking_section(prediction)
        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]
        rewards = []
        if len(prediction) == 0 or len(answer) == 0:
            logger.warning("Empty prediction received for IFEvalVerifier.")
            return VerificationResult(score=0.0)
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if prediction.strip() and instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return VerificationResult(score=sum(rewards) / len(rewards))
    
class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)


class MathVerifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []

        # Attempt extraction from \boxed{}.
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        # Attempt extraction via Minerva format.
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Attempt extraction from the last LaTeX-formatted answer.
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)

        # Fallback to the full output.
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
            # also provide original string in case normalization fails
            all_answers.append(prediction)

        # Compare each candidate answer to the ground truth.
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)