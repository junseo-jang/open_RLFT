import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, concatenate_datasets
import argparse
import wandb
from open_instruct.my_utils import IFEvalVerifier

# ============================================================
# RLVR-IFeval용: func_name → instruction_id 매핑
# (이 데이터셋은 func_name 체계를 사용)
# ============================================================

def _quantifier_to_relation(quantifier):
    """RLVR-IFeval quantifier → Google IFEval relation 변환."""
    if quantifier is None:
        return None
    q = quantifier.lower().strip()
    if q in ("at least", "around"):
        return "at least"
    elif q in ("at most", "less than"):
        return "less than"
    return "at least"


_ID = lambda gt: {}

FUNC_NAME_TO_INSTRUCTION = {
    # --- 파라미터 없는 checker ---
    "validate_lowercase":      ("change_case:english_lowercase",              _ID),
    "validate_uppercase":      ("change_case:english_capital",                _ID),
    "validate_no_commas":      ("punctuation:no_comma",                       _ID),
    "validate_quotation":      ("startend:quotation",                         _ID),
    "validate_json_format":    ("detectable_format:json_format",              _ID),
    "validate_title":          ("detectable_format:title",                    _ID),
    "validate_two_responses":  ("combination:two_responses",                  _ID),
    "validate_choice":         ("detectable_format:constrained_response",     _ID),
    # --- 파라미터 있는 checker ---
    "validate_end": (
        "startend:end_checker",
        lambda gt: {"end_phrase": gt.get("end_phrase")},
    ),
    "validate_forbidden_words": (
        "keywords:forbidden_words",
        lambda gt: {"forbidden_words": gt.get("forbidden_words")},
    ),
    "validate_highlighted_sections": (
        "detectable_format:number_highlighted_sections",
        lambda gt: {"num_highlights": gt.get("N")},
    ),
    "validate_placeholders": (
        "detectable_content:number_placeholders",
        lambda gt: {"num_placeholders": gt.get("N")},
    ),
    "validate_repeat_prompt": (
        "combination:repeat_prompt",
        lambda gt: {"prompt_to_repeat": gt.get("original_prompt")},
    ),
    "validate_sections": (
        "detectable_format:multiple_sections",
        lambda gt: {"section_spliter": gt.get("section_splitter"), "num_sections": gt.get("N")},
    ),
    "validate_paragraphs": (
        "length_constraints:nth_paragraph_first_word",
        lambda gt: {"num_paragraphs": gt.get("N"), "nth_paragraph": gt.get("i"), "first_word": gt.get("first_word")},
    ),
    "validate_frequency_capital_words": (
        "change_case:capital_word_frequency",
        lambda gt: {"capital_frequency": gt.get("N"), "capital_relation": _quantifier_to_relation(gt.get("quantifier"))},
    ),
    "validate_word_constraint": (
        "length_constraints:number_words",
        lambda gt: {"num_words": gt.get("N"), "relation": _quantifier_to_relation(gt.get("quantifier"))},
    ),
    "verify_paragraph_count": (
        "length_constraints:number_paragraphs",
        lambda gt: {"num_paragraphs": gt.get("N")},
    ),
    "verify_sentence_constraint": (
        "length_constraints:number_sentences",
        lambda gt: {"num_sentences": gt.get("N"), "relation": _quantifier_to_relation(gt.get("quantifier"))},
    ),
    "verify_keywords": (
        "keywords:existence",
        lambda gt: {"keywords": gt.get("keyword_list")},
    ),
    "verify_keyword_frequency": (
        "keywords:frequency",
        lambda gt: {"keyword": gt.get("word"), "frequency": gt.get("N"), "relation": "at least"},
    ),
    "verify_letter_frequency": (
        "keywords:letter_frequency",
        lambda gt: {"letter": gt.get("letter"), "let_frequency": gt.get("N"), "let_relation": "at least"},
    ),
    "verify_bullet_points": (
        "detectable_format:number_bullet_lists",
        lambda gt: {"num_bullets": gt.get("N")},
    ),
    "verify_postscript": (
        "detectable_content:postscript",
        lambda gt: {"postscript_marker": gt.get("postscript_marker")},
    ),
}


# ============================================================
# Dataset 전처리
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["messages"][0]["content"]},
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="OpenRLFT")
    parser.add_argument("--wandb_name", type=str, default="if-grpo")
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--use_vllm", type=bool, default=True)
    parser.add_argument("--vllm_mode", type=str, default="server")
    parser.add_argument("--vllm_model_impl", type=str, default="vllm")
    parser.add_argument("--beta", type=float, default=0.01)

    args = parser.parse_args()
    model_name = args.model_name
    output_dir = args.output_dir
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    per_device_train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    max_completion_length = args.max_completion_length
    num_generations = args.num_generations
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    push_to_hub = args.push_to_hub
    save_strategy = args.save_strategy
    save_steps = args.save_steps
    use_vllm = args.use_vllm
    vllm_mode = args.vllm_mode
    vllm_model_impl = args.vllm_model_impl
    logging_steps = args.logging_steps
    beta = args.beta
    verifier = IFEvalVerifier()
    # --- 데이터셋 로드 ---
    ds1_id = "allenai/RLVR-IFeval"
    ds2_id = "allenai/IF_multi_constraints_upto5"

    ds1_train = load_dataset(ds1_id, split="train")
    ds2_train = load_dataset(ds2_id, split="train")

    ds1_train = ds1_train.map(make_conversation)
    ds2_train = ds2_train.map(make_conversation)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{%- if tools %}", "{%- if false %}"
    )
    # 공통 컬럼만 남기고 합치기
    common_cols = ["prompt", "ground_truth"]
    ds1_train = ds1_train.select_columns(common_cols)
    ds2_train = ds2_train.select_columns(common_cols)
    train_dataset = concatenate_datasets([ds1_train, ds2_train]).shuffle(seed=42)
        # 극단적으로 긴 프롬프트 제거
    def filter_long_prompts(example):
        content = example["prompt"][-1]["content"]  # user message
        return len(tokenizer.encode(content, add_special_tokens=False)) <= 4096

    train_dataset = train_dataset.filter(filter_long_prompts, num_proc=32)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,

        # Data preprocessing
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_train_batch_size,

        # Reporting and saving
        report_to=["wandb"],
        run_name=wandb_name,

        push_to_hub=push_to_hub,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        # vLLM
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_model_impl=vllm_model_impl,
        vllm_server_base_url="http://localhost:8000",
        chat_template_kwargs={"enable_thinking": False},

        beta=beta
        )

    # 두 데이터셋 합쳤으므로 통합 reward 사용
    # ground_truth 형식으로 자동 분기: JSON이면 RLVR, Python literal이면 multi
    def unified_reward(completions, **kwargs):
        """두 데이터셋의 ground_truth 형식에 따라 자동 분기."""
        ground_truths = kwargs["ground_truth"]
        completion_contents = [c[0]["content"] for c in completions]
        rewards = []
        for content, gt_str in zip(completion_contents, ground_truths):
            try:
                # RLVR-IFeval: JSON dict with "func_name" → instruction_id 포맷으로 변환
                gt = json.loads(gt_str)
                if isinstance(gt, dict) and "func_name" in gt:
                    func_name = gt["func_name"]
                    if func_name not in FUNC_NAME_TO_INSTRUCTION:
                        rewards.append(0.0)
                        continue
                    instruction_id, param_fn = FUNC_NAME_TO_INSTRUCTION[func_name]
                    label = str([{"instruction_id": [instruction_id], "kwargs": [param_fn(gt)]}])
                    result = verifier(tokenized_prediction=[], prediction=content, label=label)
                    rewards.append(result.score)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # IF_multi_constraints: IFEvalVerifier가 직접 처리
            try:
                result = verifier(tokenized_prediction=[], prediction=content, label=gt_str)
                rewards.append(result.score)
            except Exception:
                rewards.append(0.0)

        return rewards

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[unified_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(dataset_name=f"{ds1_id}+{ds2_id}", repo_id=output_dir)