from unsloth import FastLanguageModel
import torch

# 🚀 SEPHLIGHTY AI: GPU TRAINING SCRIPT
# This script uses your GPU to make the AI smart.

def train_sephlighty_ai():
    print("--- STARTING GPU TRAINING ---")
    
    # 1. Load the Base Model (Llama 3 8B)
    # 4-bit loading makes it faster and uses less VRAM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters (The "Fine-Tuning" Magic)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None, 
    )

    # 3. Load Your Data
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="sephlighty_training_data.jsonl", split="train")

    # 4. Train!
    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Increase this for better results (e.g., 500)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer.train()

    # 5. Save the Model (GGUF Format for OmniBrain)
    print("✅ Training Complete! Saving model...")
    model.save_pretrained("sephlighty_model_v1")
    model.save_pretrained_gguf("sephlighty_model_gguf", tokenizer, quantization_method = "q4_k_m")

if __name__ == "__main__":
    train_sephlighty_ai()
