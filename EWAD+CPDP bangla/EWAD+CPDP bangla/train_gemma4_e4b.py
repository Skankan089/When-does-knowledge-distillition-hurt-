"""
Fine-tuning script for Gemma 4 E2B Model
============================================================
Model  : google/gemma-4-E2B
Task   : Bengali Abstractive Summarization (BanSum ≤1000 tokens)
Method : LoRA (PEFT) + SFTTrainer (TRL) + FP16
Metrics: ROUGE-1/2/L, BERTScore, Semantic Similarity, BLEU

Requirements (install before running):
    pip install transformers>=4.51.0 trl>=0.8.6 peft>=0.10.0 accelerate>=0.28.0
    pip install rouge-score bert-score sentence-transformers sacrebleu nltk
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage:
    python train_gemma4_e4b.py
    python train_gemma4_e4b.py --skip_train   # evaluation only (load saved model)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Fix PyTorch memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Avoid tokenizer parallelism warning inside DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from rouge_score import rouge_scorer as rouge_scorer_lib


# ============================================================================
# PEFT COMPATIBILITY PATCH — Gemma4ClippableLinear
# ============================================================================
def _unwrap_gemma4_clippable_linears(model):
    """
    Gemma 4 wraps its linear projections in Gemma4ClippableLinear (soft-cap
    regularisation). PEFT LoRA only recognises plain nn.Linear, so we replace
    each wrapper with its inner nn.Linear before injecting adapters.
    The weights are unchanged; only the clipping forward-pass is removed,
    which is acceptable for supervised fine-tuning.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except ImportError:
        return model   # not needed on this transformers version

    replaced = 0
    # Collect (parent, attr_name, child) tuples first to avoid mutation during iteration
    replacements = []
    for full_name, module in model.named_modules():
        if isinstance(module, Gemma4ClippableLinear):
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], module.linear))

    for parent, attr, linear in replacements:
        setattr(parent, attr, linear)
        replaced += 1

    if replaced:
        print(f"  [patch] Replaced {replaced} Gemma4ClippableLinear -> nn.Linear for PEFT compatibility")
    return model

# ============================================================================
# CONFIGURATION
# ============================================================================

# ---- Model ----
MODEL_NAME = "google/gemma-4-E2B"

# ---- Dataset ----
BANSUM_FILE = "bansum_lte_1000_tokens.json"   # relative to script location
DATASET_TEXT_KEY = "main"
DATASET_SUMMARY_KEY = "sum2"

# ---- Splits ----
TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
SEED        = 42

# ---- Tokenisation ----
MAX_SEQ_LENGTH      = 512    # total prompt + completion tokens
MAX_NEW_TOKENS      = 256    # maximum tokens to generate during inference

# ---- LoRA ----
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---- Training ----
OUTPUT_DIR                = f"./gemma4_e2b_bansum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
NUM_TRAIN_EPOCHS          = 3
PER_DEVICE_TRAIN_BATCH    = 2
PER_DEVICE_EVAL_BATCH     = 2
GRADIENT_ACCUMULATION     = 4       # effective batch = 8
LEARNING_RATE             = 2e-4
WARMUP_RATIO              = 0.05
WEIGHT_DECAY              = 0.01
LR_SCHEDULER              = "cosine"
FP16                      = False   # disabled — Gemma 4 params are BFloat16
BF16                      = True    # BF16 has wider range; no grad scaler needed

# ---- 4-bit QLoRA quantization ----
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,   # nested quant saves ~0.4 bits/param extra
)
LOGGING_STEPS             = 50
SAVE_STEPS                = 2000
EVAL_STEPS                = 2000
SAVE_TOTAL_LIMIT          = 2
LOAD_BEST_MODEL_AT_END    = True
METRIC_FOR_BEST_MODEL     = "eval_loss"

# ---- Inference batch size (for evaluation on test set) ----
EVAL_GEN_BATCH_SIZE = 4

# ---- Chat template for base (non-IT) Gemma 4 tokenizer ----
# Mirrors the template used by the Gemma 4 instruction-tuned variants.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)

# ---- BERTScore / Semantic Similarity model ----
# Using multilingual model to support Bangla text
BERTSCORE_MODEL = "bert-base-multilingual-cased"
STSB_MODEL      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================================
# HELPER: Bangla-aware ROUGE tokenizer
# ============================================================================
class SpaceTokenizer:
    """Whitespace tokenizer for ROUGE on Bangla/CJK text."""
    def tokenize(self, text: str):
        return text.split()


# ============================================================================
# 1.  DATASET LOADING
# ============================================================================

def load_bansum(path: str) -> DatasetDict:
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame([
        {"text": item[DATASET_TEXT_KEY], "summary": item[DATASET_SUMMARY_KEY]}
        for item in raw
        if DATASET_TEXT_KEY in item and DATASET_SUMMARY_KEY in item
    ])
    print(f"Total samples loaded : {len(df)}")

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n          = len(df)
    n_train    = int(TRAIN_SPLIT * n)
    n_val      = int(VAL_SPLIT  * n)

    train_df   = df[:n_train]
    val_df     = df[n_train : n_train + n_val]
    test_df    = df[n_train + n_val :]

    print(f"Train : {len(train_df)}")
    print(f"Val   : {len(val_df)}")
    print(f"Test  : {len(test_df)}")

    return DatasetDict({
        "train"      : Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation" : Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test"       : Dataset.from_pandas(test_df.reset_index(drop=True)),
    })


# ============================================================================
# 2.  PROMPT FORMATTING  (Gemma chat template)
# ============================================================================

SYSTEM_MSG = (
    "আপনি একজন বাংলা ভাষার সংক্ষেপকারী সহকারী। "
    "নিচের বাংলা নিবন্ধটি পড়ুন এবং একটি সংক্ষিপ্ত সারসংক্ষেপ লিখুন।"
)

USER_PROMPT_TEMPLATE = (
    "{system}\n\nনিবন্ধ:\n{article}\n\nসারসংক্ষেপ:"
)


def build_user_content(article: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        system=SYSTEM_MSG,
        article=article.strip(),
    )


def build_prompt_only(article: str) -> str:
    """Build the raw prompt string used during inference."""
    return (
        f"<start_of_turn>user\n"
        f"{build_user_content(article)}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def format_for_sft(example: dict) -> dict:
    """
    Return a 'messages' list (conversation format).
    TRL 0.29 SFTTrainer with completion_only_loss=True will apply the
    tokenizer's chat template and mask out all non-assistant turns.
    """
    return {
        "messages": [
            {"role": "user",      "content": build_user_content(example["text"])},
            {"role": "assistant", "content": example["summary"].strip()},
        ]
    }


# ============================================================================
# 2b. METRICS HELPERS  (used during training evaluation)
# ============================================================================

def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax token IDs before storing — avoids OOM."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tokenizer):
    """Return a compute_metrics fn that reports ROUGE-1/2/L on completions."""
    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=SpaceTokenizer(),
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds, decoded_refs = [], []

        for pred, label in zip(predictions, labels):
            # Only score the completion tokens (label != -100)
            mask = label != -100
            pred_tokens = pred[mask]
            ref_tokens  = label[mask]
            decoded_preds.append(
                tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            )
            decoded_refs.append(
                tokenizer.decode(ref_tokens,  skip_special_tokens=True).strip()
            )

        r1, r2, rl = [], [], []
        for pred, ref in zip(decoded_preds, decoded_refs):
            if not pred or not ref:
                r1.append(0.0); r2.append(0.0); rl.append(0.0)
                continue
            s = scorer.score(ref, pred)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)

        return {
            "rouge1": float(np.mean(r1)),
            "rouge2": float(np.mean(r2)),
            "rougeL": float(np.mean(rl)),
        }

    return compute_metrics


# ============================================================================
# 3.  MODEL & TOKENIZER INITIALISATION
# ============================================================================

def load_model_and_tokenizer(model_name: str, use_lora: bool = True):
    print("\n" + "=" * 70)
    print(f"LOADING MODEL  :  {model_name}")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # Gemma base model: inject chat template and pad token if missing
    if processor.tokenizer.chat_template is None:
        processor.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        print("  [patch] Injected Gemma chat template onto tokenizer")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token     = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id  = processor.tokenizer.eos_token_id

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config = BNB_CONFIG,   # 4-bit NF4 QLoRA
        device_map          = {"": "cuda:0"},
        trust_remote_code   = True,
        attn_implementation = "eager",
    )
    model.config.use_cache = False    # required for gradient checkpointing

    if use_lora:
        # Unwrap Gemma4-specific linear wrappers so PEFT can inject LoRA
        model = _unwrap_gemma4_clippable_linears(model)

        lora_cfg = LoraConfig(
            task_type         = TaskType.CAUSAL_LM,
            r                 = LORA_RANK,
            lora_alpha        = LORA_ALPHA,
            lora_dropout      = LORA_DROPOUT,
            target_modules    = LORA_TARGET_MODULES,
            bias              = "none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, processor


# ============================================================================
# 4.  TRAINING
# ============================================================================

def train(dataset: DatasetDict, output_dir: str, resume_from: str = None):
    model, processor = load_model_and_tokenizer(MODEL_NAME, use_lora=True)

    # Format datasets into conversation format
    train_ds = dataset["train"].map(format_for_sft, remove_columns=dataset["train"].column_names)
    val_ds   = dataset["validation"].map(format_for_sft, remove_columns=dataset["validation"].column_names)

    sft_config = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size  = PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = FP16,
        bf16                        = BF16,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = SAVE_TOTAL_LIMIT,
        load_best_model_at_end      = False,          # disabled — no mid-training eval
        eval_strategy               = "no",           # evals only at the end (generate_summaries)
        save_strategy               = "steps",
        seed                        = SEED,
        report_to                   = "none",
        max_length                  = MAX_SEQ_LENGTH,
        # TRL 0.29: use conversation format + completion_only_loss
        completion_only_loss        = True,           # mask prompt tokens from loss
        packing                     = False,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model            = model,
        args             = sft_config,
        train_dataset    = train_ds,
        processing_class = processor.tokenizer,
    )

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final merged model
    final_dir = os.path.join(output_dir, "final_model")
    print(f"\nSaving final model to {final_dir} ...")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    # Save train results
    train_results = trainer.state.log_history
    with open(os.path.join(output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2, ensure_ascii=False)

    print("Training complete.")
    return final_dir, processor


# ============================================================================
# 5.  GENERATION (inference on test set)
# ============================================================================

def generate_summaries(
    model_dir: str,
    test_dataset: Dataset,
    processor,
    use_lora: bool = True,
):
    print("\n" + "=" * 70)
    print("GENERATING SUMMARIES ON TEST SET")
    print("=" * 70)

    if use_lora:
        base_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            quantization_config = BNB_CONFIG,   # 4-bit — no dequant needed for inference
            device_map          = {"":"cuda:0"},
            trust_remote_code   = True,
            attn_implementation = "eager",
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        # Keep adapter on quantized base (avoids costly dequant merge)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            dtype=torch.float16,
            device_map={"":"cuda:0"},
            trust_remote_code=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device

    predictions = []
    references  = test_dataset["summary"]
    articles    = test_dataset["text"]

    for i in range(0, len(articles), EVAL_GEN_BATCH_SIZE):
        batch_articles = articles[i : i + EVAL_GEN_BATCH_SIZE]
        prompts        = [build_prompt_only(a) for a in batch_articles]

        encodings = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens      = MAX_NEW_TOKENS,
                do_sample           = False,    # greedy decoding for reproducibility
                temperature         = 1.0,
                repetition_penalty  = 1.1,
                pad_token_id        = processor.tokenizer.pad_token_id,
                eos_token_id        = processor.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude prompt)
        input_len = encodings["input_ids"].shape[1]
        for out in output_ids:
            new_tokens = out[input_len:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            # Strip any residual turn markers
            text = text.replace("<end_of_turn>", "").strip()
            predictions.append(text)

        if (i // EVAL_GEN_BATCH_SIZE) % 10 == 0:
            done = min(i + EVAL_GEN_BATCH_SIZE, len(articles))
            print(f"  Generated {done}/{len(articles)}")

    return predictions, list(references)


# ============================================================================
# 6.  METRICS
# ============================================================================

def compute_rouge(predictions, references):
    print("\nComputing ROUGE scores ...")
    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        tokenizer=SpaceTokenizer(),
    )
    r1_list, r2_list, rl_list = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
    return {
        "rouge1" : float(np.mean(r1_list)),
        "rouge2" : float(np.mean(r2_list)),
        "rougeL" : float(np.mean(rl_list)),
    }


def compute_bleu(predictions, references):
    print("Computing BLEU score ...")
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="char")
        return {"bleu": bleu.score / 100.0}
    except ImportError:
        pass

    # Fallback: NLTK sentence-level BLEU
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        if not pred_tokens:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoother))
    return {"bleu": float(np.mean(scores))}


def compute_bertscore(predictions, references):
    print("Computing BERTScore ...")
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type  = BERTSCORE_MODEL,
        lang        = "bn",
        verbose     = False,
        batch_size  = 32,
    )
    return {
        "bertscore_precision" : float(P.mean()),
        "bertscore_recall"    : float(R.mean()),
        "bertscore_f1"        : float(F1.mean()),
    }


def compute_semantic_similarity(predictions, references):
    print("Computing Semantic Similarity ...")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    stsb = SentenceTransformer(STSB_MODEL)
    pred_embs = stsb.encode(predictions,  batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    ref_embs  = stsb.encode(references,   batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    sims = cosine_similarity(pred_embs, ref_embs).diagonal()
    return {"semantic_similarity": float(np.mean(sims))}


def evaluate_all(predictions, references, output_dir: str):
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)

    results = {}
    results.update(compute_rouge(predictions, references))
    results.update(compute_bleu(predictions, references))
    results.update(compute_bertscore(predictions, references))
    results.update(compute_semantic_similarity(predictions, references))

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"  {k:<30s}: {v:.4f}")

    # Save results
    out_path = os.path.join(output_dir, "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {out_path}")

    # Save predictions
    pred_path = os.path.join(output_dir, "test_predictions.json")
    pred_data = [
        {"article": a, "reference": r, "prediction": p}
        for a, r, p in zip(
            list(references)[:len(predictions)],
            references,
            predictions,
        )
    ]
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {pred_path}")

    return results


# ============================================================================
# 7.  MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 E2B on BanSum")
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and only run evaluation (requires --model_dir)",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to a previously trained model directory (for skip_train mode)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=BANSUM_FILE,
        help=f"Path to BanSum JSON file (default: {BANSUM_FILE})",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from (e.g. ./gemma4_e2b_bansum_.../checkpoint-14000)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dataset path relative to script location
    script_dir   = Path(__file__).parent
    dataset_path = args.dataset if os.path.isabs(args.dataset) else str(script_dir / args.dataset)

    print("\n" + "=" * 70)
    print("  Gemma 4 E2B  |  Bengali Summarisation  |  FP16 + LoRA")
    print("=" * 70)
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Dataset        : {dataset_path}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print(f"  FP16           : {FP16}")
    print(f"  LoRA rank/alpha: {LORA_RANK}/{LORA_ALPHA}")
    print(f"  Epochs         : {NUM_TRAIN_EPOCHS}")
    print(f"  Batch size     : {PER_DEVICE_TRAIN_BATCH} x {GRADIENT_ACCUMULATION} accum = {PER_DEVICE_TRAIN_BATCH * GRADIENT_ACCUMULATION} effective")
    print(f"  Max seq length : {MAX_SEQ_LENGTH}")

    # Load dataset
    dataset = load_bansum(dataset_path)

    # --- Training ---
    if args.skip_train:
        if args.model_dir is None:
            print("ERROR: --model_dir must be provided when using --skip_train")
            sys.exit(1)
        final_model_dir = args.model_dir
        processor = AutoProcessor.from_pretrained(final_model_dir, trust_remote_code=True)
        if processor.tokenizer.chat_template is None:
            processor.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token    = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        output_dir = os.path.dirname(final_model_dir)
    else:
        output_dir      = OUTPUT_DIR if args.resume is None else str(Path(args.resume).parent)
        os.makedirs(output_dir, exist_ok=True)
        final_model_dir, processor = train(dataset, output_dir, resume_from=args.resume)

    # --- Inference ---
    predictions, references = generate_summaries(
        model_dir   = final_model_dir,
        test_dataset= dataset["test"],
        processor   = processor,
        use_lora    = not args.skip_train,   # merge LoRA adapter if we just trained
    )

    # --- Evaluation ---
    evaluate_all(predictions, references, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
