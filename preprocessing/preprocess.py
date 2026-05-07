import re
import pandas as pd
from collections import OrderedDict


class Preprocessor:
    """
    Generic preprocessing module used in log parsers such as
    Drain, Spell, IPLoM, etc.

    Functions:
    1. Extract content field
    2. Apply regex-based variable substitution
    3. Remove duplicate logs
    4. Preserve original indices
    """

    def __init__(self, rex=None):
        """
        rex : list of regex patterns
        """
        self.rex = rex if rex else []

    # --------------------------------------------------------
    # Step 1: Extract Content Field
    # --------------------------------------------------------
    def extract_content(self, log_line):
        """
        Extract only message content from raw log line.

        Modify this function according to dataset format.
        """

        # Example for HDFS-style logs
        parts = log_line.strip().split()

        # Keep content after first 5 fields
        if len(parts) > 5:
            content = " ".join(parts[5:])
        else:
            content = log_line.strip()

        return content

    # --------------------------------------------------------
    # Step 2: Regex Variable Substitution
    # --------------------------------------------------------
    def normalize(self, log):
        """
        Replace dynamic variables with <*> wildcard.
        """

        for current_rex in self.rex:
            log = re.sub(current_rex, "<*>", log)

        return log

    # --------------------------------------------------------
    # Step 3: Full Preprocessing Pipeline
    # --------------------------------------------------------
    def preprocess_logs(self, log_file):
        """
        Returns:
            unique_logs      : list of unique processed logs
            index_mapping    : mapping to original indices
        """

        processed_logs = []
        index_mapping = OrderedDict()

        with open(log_file, "r", encoding="utf-8") as f:

            for idx, line in enumerate(f):

                # Extract content
                content = self.extract_content(line)

                # Normalize variables
                normalized = self.normalize(content)

                processed_logs.append(normalized)

                # Preserve original index
                if normalized not in index_mapping:
                    index_mapping[normalized] = []

                index_mapping[normalized].append(idx)

        # Remove duplicates
        unique_logs = list(index_mapping.keys())

        return unique_logs, index_mapping

    # --------------------------------------------------------
    # Step 4: Save Processed Logs
    # --------------------------------------------------------
    def save_processed_logs(self, unique_logs, output_file):

        df = pd.DataFrame({
            "ProcessedLog": unique_logs
        })

        df.to_csv(output_file, index=False)

        print(f"[INFO] Saved processed logs to: {output_file}")


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":

    # Common regex used in log parsers
    regex_list = [
        r'blk_-?\d+',                    # block IDs
        r'(\d+\.){3}\d+(:\d+)?',         # IP addresses
        r'\b\d+\b'                       # numbers
    ]

    processor = Preprocessor(rex=regex_list)

    log_file = "datasets/HDFS/HDFS.log"

    unique_logs, index_mapping = processor.preprocess_logs(log_file)

    print(f"Total Unique Logs: {len(unique_logs)}")

    processor.save_processed_logs(
        unique_logs,
        "datasets/HDFS/HDFS_processed.csv"
    )

    # Example mapping
    sample_key = list(index_mapping.keys())[0]

    print("\nExample Processed Log:")
    print(sample_key)

    print("\nOriginal Line Indices:")
    print(index_mapping[sample_key][:10])
