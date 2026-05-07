import pandas as pd
import numpy as np
import torch

from transformers import BertTokenizer, BertModel


class BertEmbedder:

    def __init__(self,
                 model_name='bert-base-uncased',
                 max_length=64):

        print("[INFO] Loading BERT model...")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        self.max_length = max_length

        # evaluation mode
        self.model.eval()

        print("[INFO] BERT loaded successfully.")

    # --------------------------------------------------------
    # Generate embedding for one log message
    # --------------------------------------------------------
    def get_embedding(self, log_message):

        # Tokenize input log
        inputs = self.tokenizer(
            log_message,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        # Disable gradient computation
        with torch.no_grad():

            outputs = self.model(**inputs)

        # Last hidden states
        token_embeddings = outputs.last_hidden_state

        # Shape:
        # [1, seq_len, 768]

        # Sum pooling
        log_vector = torch.sum(token_embeddings, dim=1)

        # Convert tensor to numpy
        log_vector = log_vector.squeeze().numpy()

        return log_vector

    # --------------------------------------------------------
    # Generate embeddings for all logs
    # --------------------------------------------------------
    def generate_embeddings(self, input_csv):

        df = pd.read_csv(input_csv)

        logs = df["ProcessedLog"].tolist()

        embeddings = []

        print(f"[INFO] Generating embeddings for {len(logs)} logs...\n")

        for idx, log in enumerate(logs):

            vector = self.get_embedding(log)

            embeddings.append(vector)

            if (idx + 1) % 10 == 0:
                print(f"[INFO] Processed {idx + 1}/{len(logs)} logs")

        embeddings = np.array(embeddings)

        print("\n[INFO] Embedding generation completed.")
        print(f"[INFO] Embedding shape: {embeddings.shape}")

        return logs, embeddings

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------
    def save_embeddings(self,
                        embeddings,
                        output_file):

        np.save(output_file, embeddings)

        print(f"[INFO] Saved embeddings to: {output_file}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    input_csv = "datasets/HDFS/HDFS_processed.csv"

    output_file = "datasets/HDFS/HDFS_embeddings.npy"

    embedder = BertEmbedder()

    logs, embeddings = embedder.generate_embeddings(input_csv)

    embedder.save_embeddings(
        embeddings,
        output_file
    )

    # Example output
    print("\nExample Log:")
    print(logs[0])

    print("\nEmbedding Vector Shape:")
    print(embeddings[0].shape)
