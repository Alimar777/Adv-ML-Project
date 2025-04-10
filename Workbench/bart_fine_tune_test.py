import os
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

# Custom dataset class
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

# Prepare dataset from transition_sheet.csv
def prepare_dataset(model_name="facebook/bart-base"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "transition_sheet.csv")

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

    train_dataset = SceneDifferenceDataset(train_inputs, train_labels)
    val_dataset = SceneDifferenceDataset(val_inputs, val_labels)

    # Save test dataframe
    df_without_path = os.path.join(base_dir, "unlabeled_transitions.csv")
    df_without_transition.to_csv(df_without_path, index=False)

    return train_dataset, val_dataset, tokenizer

# Fine-tune BART model
def fine_tune_bart():
    model_name = "facebook/bart-base"
    train_dataset, val_dataset, tokenizer = prepare_dataset(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained("./bart-scene-difference-detector")
    tokenizer.save_pretrained("./bart-scene-difference-detector")

    return model, tokenizer

# Generate and save transitions for missing examples
def test_model(model, tokenizer):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv_path = os.path.join(base_dir, "unlabeled_transitions.csv")
    df = pd.read_csv(test_csv_path)
    df = df.dropna(subset=["prev_caption", "curr_caption"])

    df["input_text"] = df.apply(
        lambda row: f"Caption 1: {row['prev_caption']}\nCaption 2: {row['curr_caption']}\nWhat's the transition?",
        axis=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    generated_transitions = []

    for _, row in df.iterrows():
        input_text = row["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        output = model.generate(
            inputs["input_ids"],
            max_length=32,
            min_length=5,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True
        )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_transitions.append(decoded_output)

    df["generated_transition"] = generated_transitions
    out_path = os.path.join(base_dir, "generated_transitions.csv")
    df.to_csv(out_path, index=False)
    print(f"\nGenerated transitions saved to: {out_path}")

# Main
if __name__ == "__main__":
    print("Starting BART fine-tuning for transition generation...")
    model, tokenizer = fine_tune_bart()

    print("\nGenerating transitions for unlabeled examples...")
    test_model(model, tokenizer)

    print("\nAll done! Model and outputs saved.")
