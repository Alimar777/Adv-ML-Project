from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

'''captions = [
    "a man walking down a dirt path with red flowers",
    "a man walking down a dirt path with a backpack",
    "a man walking down a dirt path with a backpack",
    "a person walking down a dirt path with red flowers",
    "a person walking down a dirt path with red flowers",
    "a man walking down a dirt path with a backpack",
    "a man walking down a dirt path with a backpack",
    "a person walking down a dirt path with red flowers",
    "a man walking down a dirt path with red flowers",
    "a person walking down a dirt road with red flowers",
    "a person walking down a dirt road with red flowers",
    "a person walking down a dirt path with red flowers"
]'''

captions = [
    "a woman is making flowers in a vase",
    "a woman arranging flowers in a flower shop",
    "a woman is holding a flower in her hand",
    "a woman is arranging flowers in a vase",
    "a woman is arranging flowers in a vase",
    "a woman is arranging flowers in a vase",
    "a woman is putting a flower in a vase",
    "a woman is sitting at a table with flowers",
    "a woman is cutting flowers on a table",
    "a woman cutting a flower with scissors",
    "a woman is cutting flowers on a table",
    "a woman is cutting a piece of food",
    "a woman is cutting a flower with scissors",
    "a woman is cutting a flower with scissors",
    "a woman is cutting flowers on a table",
    "a woman is cutting a flower with scissors",
    "a woman is cutting a flower with scissors",
    "a woman is cutting flowers on a table",
    "a woman is cutting a flower with scissors",
    "a woman cutting a flower with scissors",
    "a woman is cutting a flower with scissors",
    "a woman is making flowers in a vase"
]


input_text = "Summarize the following descriptions into a single coherent scene:\n" + "\n".join(captions)

models = {
    "FLAN-T5": "google/flan-t5-base",
    "DistilBART": "sshleifer/distilbart-cnn-12-6"
}

results = {}

for name, model_id in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    summary = summarizer(input_text, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
    results[name] = summary

# Print side-by-side results
print("\n=== Summarization Comparison ===")
for name, summary in results.items():
    print(f"{name}:\n  {summary}\n")
