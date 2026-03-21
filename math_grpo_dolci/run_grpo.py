import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, concatenate_datasets
import argparse
import wandb
# from nemo_skills.evaluation.math_grader import extract_answer, math_equal
from open_instruct.my_utils import MathVerifier

# ============================================================
# Dataset 전처리
# ============================================================

SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions carefully."
MATH_SOURCES = {
    "hamishivi/omega-combined-no-boxed_filtered",
    "hamishivi/rlvr_orz_math_57k_collected_filtered",
    "hamishivi/MathSub-30K_filtered",
    "hamishivi/DAPO-Math-17k-Processed_filtered",
    "hamishivi/polaris_53k",
}


def make_conversation(example):
    
    raw_prompt = example["prompt"]
    # "user: ..." 형식에서 "user: " 이후 텍스트만 추출
    if raw_prompt.startswith("user:"):
        raw_prompt = raw_prompt[len("user:"):].lstrip()
    user_prompt = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}. \n\n" + raw_prompt
    
    return {
        
        # "prompt": [
        #     {"role": "system", "content": SYSTEM_PROMPT},
        #     {"role": "user", "content": raw_prompt},
        # ],
        
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
    parser.add_argument("--wandb_name", type=str, default="math-dolci")
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
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
    # --- 데이터셋 로드 ---
    ds1_id = "allenai/Dolci-Instruct-RL"

    ds1_train = load_dataset(ds1_id, split="train")

    ds1_train = ds1_train.filter(lambda x: x["dataset_source"] in MATH_SOURCES)
    ds1_train = ds1_train.map(make_conversation)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "{%- if tools %}", "{%- if false %}"
    )
    # 공통 컬럼만 남기고 합치기
    common_cols = ["prompt", "ground_truth", "dataset"]
    ds1_train = ds1_train.select_columns(common_cols)
    train_dataset = ds1_train.shuffle(seed=42)
        # 극단적으로 긴 프롬프트 제거
    def filter_long_prompts(example):
        content = example["prompt"][-1]["content"]  # user message
        return len(tokenizer.encode(content, add_special_tokens=False)) <= 2048

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
    math_verifier = MathVerifier()

    def math_reward(completions, **kwargs):
        ground_truths = kwargs["ground_truth"][0]
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            content = completion[0]["content"]
            result = math_verifier([], content, gt)
            rewards.append(result.score)
        return rewards


    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[math_reward],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if push_to_hub:
        trainer.push_to_hub(dataset_name=f"{ds1_id}+{ds2_id}", repo_id=output_dir)
