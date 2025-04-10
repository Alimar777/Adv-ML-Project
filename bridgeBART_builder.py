import os
import torch
import random
import pandas as pd
import numpy as np
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

def run_bridgebart_training_pipeline(force_train=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Models", "bridgeBART")
    transition_sheet_path = os.path.join(base_dir, "Training Data", "transition_sheet.csv")
    
    if not os.path.exists(transition_sheet_path):
        raise FileNotFoundError(f"Required file not found: {transition_sheet_path}")
    
    required_files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if force_train or not os.path.exists(model_path) or len(missing_files) > 0:
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        print("Training model since the following files are missing:", ', '.join(missing_files) if missing_files else "[initial run]")
        model, tokenizer, df_unlabeled = fine_tune_bart(base_dir)
        cosine_similarity_filter(df_unlabeled, top_n=10)
    else:
        print("Model already exists at", model_path)

class SceneDifferenceDataset(TorchDataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item["labels"] = self.labels["input_ids"][idx]
        return item

def prepare_dataset(base_dir, model_name="facebook/bart-base"):
    csv_path = os.path.join(base_dir, "Training Data", "transition_sheet.csv")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["video", "Input 1:", "Input 2:", "Outputs:"], errors="ignore")
    df = df.dropna(subset=["prev_caption", "curr_caption"])

    df["input_text"] = df.apply(
        lambda row: f"Caption 1: {row['prev_caption']}\nCaption 2: {row['curr_caption']}\nWhat's the transition?",
        axis=1
    )

    df_with_transition = df.dropna(subset=["human_transition"])
    df_without_transition = df[df["human_transition"].isna()]

    train_df, val_df = train_test_split(df_with_transition, test_size=0.2, random_state=42)

    tokenizer = BartTokenizer.from_pretrained(model_name)

    def tokenize_dataset(dataframe):
        input_texts = dataframe["input_text"].tolist()
        target_texts = dataframe["human_transition"].tolist()

        inputs = tokenizer(
            input_texts,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tokenizer(
            target_texts,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs, labels

    train_inputs, train_labels = tokenize_dataset(train_df)
    val_inputs, val_labels = tokenize_dataset(val_df)

    return SceneDifferenceDataset(train_inputs, train_labels), SceneDifferenceDataset(val_inputs, val_labels), tokenizer, df_without_transition

def fine_tune_bart(base_dir):
    model_name = "facebook/bart-base"
    train_dataset, val_dataset, tokenizer, df_unlabeled = prepare_dataset(base_dir, model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=os.path.join(base_dir, "Models", "bridgeBART"),
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        logging_dir=os.path.join(base_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    save_path = os.path.join(base_dir, "Models", "bridgeBART")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer, df_unlabeled

def cosine_similarity_filter(df, top_n=10):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    captions = list(zip(df['prev_caption'], df['curr_caption']))
    similarities = []

    for prev, curr in captions:
        emb_prev = model.encode(prev, convert_to_tensor=True)
        emb_curr = model.encode(curr, convert_to_tensor=True)
        sim = cosine_similarity([emb_prev.cpu().numpy()], [emb_curr.cpu().numpy()])[0][0]
        similarities.append(sim)

    df["cosine_similarity"] = similarities

    # Pick 5 random pairs
    random_pairs = df.sample(n=5, random_state=42)

    # Load the fine-tuned BART model
    bart_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Models", "bridgeBART"))
    model = BartForConditionalGeneration.from_pretrained(bart_model_path)
    tokenizer = BartTokenizer.from_pretrained(bart_model_path)

    print("\nSample Output from 5 Random Caption Pairs:")
    for i, row in random_pairs.iterrows():
        input_text = f"Caption 1: {row['prev_caption']}\nCaption 2: {row['curr_caption']}\nWhat's the transition?"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=32,
                num_beams=4,
                early_stopping=True
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nInput:\n{input_text}\nGenerated Transition:\n{output_text}")
