import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="모델을 bf16으로 변환 후 Hugging Face Hub에 업로드")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="로드할 모델 디렉토리 경로")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face Hub 레포지토리 ID (예: username/model-name)")
    parser.add_argument("--private", action="store_true", default=False,
                        help="비공개 레포지토리로 업로드")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"모델 로드 중: {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )

    print(f"모델을 bf16으로 변환 중...")
    model = model.to(torch.bfloat16)

    print(f"Hugging Face Hub에 업로드 중: {args.repo_id}")

    push_kwargs = {"private": args.private}

    model.push_to_hub(args.repo_id, **push_kwargs)
    tokenizer.push_to_hub(args.repo_id, **push_kwargs)

    print("업로드 완료!")


if __name__ == "__main__":
    main()
