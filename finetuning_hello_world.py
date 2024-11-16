from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
import torch

def setup_model_and_tokenizer(model_name, max_seq_length, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


def prepare_dataset(tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def setup_trainer(model, tokenizer, dataset, max_seq_length):
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    return trainer


def generate_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )


# Configuration
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

# Setup
model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, MAX_SEQ_LENGTH)
dataset = prepare_dataset(tokenizer)
trainer = setup_trainer(model, tokenizer, dataset, MAX_SEQ_LENGTH)

# Training
trainer_stats = trainer.train()

# Prepare for inference
FastLanguageModel.for_inference(model)

# Example generation
generate_response(model, tokenizer, "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,")
