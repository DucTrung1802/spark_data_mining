from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import AdamW
import mlflow
from sacrebleu import corpus_bleu
from rouge import Rouge


spark = SparkSession.builder.appName("arXivTitles").getOrCreate()
mlflow.set_tracking_uri("http://localhost:5000")  # Set MLflow server URI
mlflow.set_experiment("arXiv_Abstract_to_Title")

# Example input for T5 model (single text input)


BATCH_SIZE = 4
PATIENCE = 5

data = (    
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv("data/ML-Arxiv-Papers.csv")
    .drop("Unnamed: 0", "_c0")
)

# Format data for T5
data = data.withColumn(
    "t5_input", concat(lit("Generate title from abstract: "), data["abstract"])
)
data = data.withColumn("t5_output", data["title"])
# Split data into train, validation, and test sets
train_data, val_data, test_data = data.randomSplit([0.8, 0.1, 0.1], seed=42)

# Convert to RDD
train_rdd = train_data.select("t5_input", "t5_output").rdd
val_rdd = val_data.select("t5_input", "t5_output").rdd
test_rdd = test_data.select("t5_input", "t5_output").rdd
test_samples = test_data.limit(10).collect()  # Giới hạn 10 mẫu
# Convert Spark Row objects to a list of dicts
test_samples = [
    {"abstract": row["t5_input"], "actual_title": row["t5_output"]}
    for row in test_samples
]

# T5 model setup
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base", device_map="auto")
device = torch.device("cuda")
# model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)


# Function to process a batch
def process_batch(model, tokenizer, batch_inputs, batch_outputs, device, train=True):
    inputs = tokenizer(
        batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    labels = tokenizer(
        batch_outputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )["input_ids"]

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), outputs.logits


# Function to train the model
def train_model(model, train_rdd, tokenizer, device):
    model.train()
    total_loss = 0
    train_predictions, train_targets = [], []

    batch = []
    for idx, row in enumerate(train_rdd.toLocalIterator()):
        batch.append((row.t5_input, row.t5_output))

        if len(batch) == BATCH_SIZE:
            batch_inputs = [t5_input for t5_input, _ in batch]
            batch_outputs = [t5_output for _, t5_output in batch]

            loss, logits = process_batch(
                model, tokenizer, batch_inputs, batch_outputs, device, train=True
            )
            total_loss += loss

            predictions = tokenizer.batch_decode(
                logits.argmax(-1), skip_special_tokens=True
            )
            train_predictions.extend(predictions)
            train_targets.extend(batch_outputs)

            batch = []  # Reset batch after processing

    # Process the last batch if remaining
    if batch:
        batch_inputs = [t5_input for t5_input, _ in batch]
        batch_outputs = [t5_output for _, t5_output in batch]

        loss, logits = process_batch(
            model, tokenizer, batch_inputs, batch_outputs, device, train=True
        )
        total_loss += loss

        predictions = tokenizer.batch_decode(
            logits.argmax(-1), skip_special_tokens=True
        )
        train_predictions.extend(predictions)
        train_targets.extend(batch_outputs)

    avg_loss = total_loss / (max(1, idx // BATCH_SIZE))  # Avoid division by zero
    return avg_loss, train_predictions, train_targets


# Function to evaluate the model (validation and testing)
def evaluate_model(model, data_rdd, tokenizer, device):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    batch = []
    for idx, row in tqdm(enumerate(data_rdd.toLocalIterator()), desc="Evaluating"):
        batch.append((row.t5_input, row.t5_output))

        if len(batch) == BATCH_SIZE:
            batch_inputs = [t5_input for t5_input, _ in batch]
            batch_outputs = [t5_output for _, t5_output in batch]

            with torch.no_grad():
                loss, logits = process_batch(
                    model, tokenizer, batch_inputs, batch_outputs, device, train=False
                )

            total_loss += loss
            predictions = tokenizer.batch_decode(
                logits.argmax(-1), skip_special_tokens=True
            )
            all_predictions.extend(predictions)
            all_targets.extend(batch_outputs)

            batch = []  # Reset batch after processing

    # Process the last batch if remaining
    if batch:
        batch_inputs = [t5_input for t5_input, _ in batch]
        batch_outputs = [t5_output for _, t5_output in batch]

        with torch.no_grad():
            loss, logits = process_batch(
                model, tokenizer, batch_inputs, batch_outputs, device, train=False
            )

        total_loss += loss
        predictions = tokenizer.batch_decode(
            logits.argmax(-1), skip_special_tokens=True
        )
        all_predictions.extend(predictions)
        all_targets.extend(batch_outputs)

    # Tính toán trung bình loss
    avg_loss = total_loss / (max(1, idx // BATCH_SIZE))  # Tránh chia cho 0

    # Tính điểm BLEU
    bleu_score = corpus_bleu(all_predictions, [all_targets]).score

    # Tính điểm ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
    rouge_scores = Rouge().get_scores(all_predictions, all_targets)

    # Tính trung bình ROUGE-1, ROUGE-2, ROUGE-L
    avg_rouge_1 = sum([score["rouge-1"]["f"] for score in rouge_scores]) / len(
        rouge_scores
    )
    avg_rouge_2 = sum([score["rouge-2"]["f"] for score in rouge_scores]) / len(
        rouge_scores
    )
    avg_rouge_l = sum([score["rouge-l"]["f"] for score in rouge_scores]) / len(
        rouge_scores
    )

    return avg_loss, bleu_score, avg_rouge_l, avg_rouge_1, avg_rouge_2


# MLflow tracking with early stopping
best_val_loss = np.inf
epochs_no_improve = 0  # Counter for early stopping

with mlflow.start_run():
    for epoch in tqdm(range(100)):  # Max epochs (will stop early if needed)
        print(f"Epoch {epoch + 1}")

        # Training phase
        train_loss, train_predictions, train_targets = train_model(
            model, train_rdd, tokenizer, device
        )
        bleu_score = corpus_bleu(train_predictions, [train_targets]).score
        rouge_scores = Rouge().get_scores(train_predictions, train_targets)
        avg_rouge_l = sum([score["rouge-l"]["f"] for score in rouge_scores]) / len(
            rouge_scores
        )
        avg_rouge_1 = sum([score["rouge-1"]["f"] for score in rouge_scores]) / len(
            rouge_scores
        )
        avg_rouge_2 = sum([score["rouge-2"]["f"] for score in rouge_scores]) / len(
            rouge_scores
        )

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_bleu": bleu_score,
                "train_rouge_l": avg_rouge_l,
                "train_rouge_1": avg_rouge_1,
                "train_rouge_2": avg_rouge_2,
            },
            step=epoch,
        )

        # Validation phase
        val_loss, val_bleu, val_rouge_l, val_rouge_1, val_rouge_2 = evaluate_model(
            model, val_rdd, tokenizer, device
        )
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "val_rouge_l": val_rouge_l,
                "val_rouge_1": val_rouge_1,
                "val_rouge_2": val_rouge_2,
            },
            step=epoch,
        )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            mlflow.pytorch.log_model(model, "best_model")
        else:
            epochs_no_improve += 1

        print(
            f"Validation Loss: {val_loss:.4f} | Best Loss: {best_val_loss:.4f} | No Improvement: {epochs_no_improve}/{PATIENCE}"
        )

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered. Stopping training.")
            break  # Stop training

    # Testing phase after training
    test_loss, test_bleu, test_rouge_l, test_rouge_1, test_rouge_2 = evaluate_model(
        model, test_rdd, tokenizer, device
    )
    mlflow.log_metrics(
        {
            "test_loss": test_loss,
            "test_bleu": test_bleu,
            "test_rouge_l": test_rouge_l,
            "test_rouge_1": test_rouge_1,
            "test_rouge_2": test_rouge_2,
        }
    )

    actual_titles = []
    generated_titles = []

    for sample in test_samples:
        abstract = sample["abstract"]
        actual_title = sample["actual_title"]

        # Tokenize input text
        input_ids = tokenizer(
            abstract, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)

        # Generate output with Beam Search
        output_ids = model.generate(
            input_ids, max_length=20, num_beams=5, early_stopping=True
        )

        # Decode generated title
        generated_title = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Store results
        generated_titles.append(
            {
                "abstract": abstract,
                "actual_title": actual_title,
                "generated_title": generated_title,
            }
        )

    # Save all generated results into a text file
    with open("test_results.txt", "w") as f:
        for entry in generated_titles:
            f.write(f"Abstract: {entry['abstract']}\n")
            f.write(f"Actual Title: {entry['actual_title']}\n")
            f.write(f"Generated Title: {entry['generated_title']}\n")
            f.write("=" * 80 + "\n")
    # Log the test results file to MLflow
    mlflow.log_artifact("test_results.txt")

    # Log the final model
    mlflow.pytorch.log_model(model, "final_model")

# Close Spark session
spark.stop()
