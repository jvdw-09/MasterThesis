import pandas as pd
import gzip
import glob
import os

# Define directories
source_dir = "/Users/jortvanderweijde/Desktop/Pythontest/Parts"
output_dir = "/Users/jortvanderweijde/Desktop/Pythontest/Part pos"
os.makedirs(output_dir, exist_ok=True)

# Function to split the dataframe into N parts with overlap
def split_dataframe(df, parts=5, overlap=6):
    chunk_size = len(df) // parts
    chunks = []
    for i in range(parts):
        start = max(0, i * chunk_size - overlap)
        end = min(len(df), (i + 1) * chunk_size + overlap)
        chunks.append(df.iloc[start:end].copy())
    return chunks

# Get all .csv and .csv.gz files
files = glob.glob(os.path.join(source_dir, "*USD*.csv")) + glob.glob(os.path.join(source_dir, "*USD*.csv.gz"))

# Process each file
for file_path in files:
    try:
        # Detect if the file is compressed
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
        else:
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)

        # Convert date column to datetime and set index
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        if 'close' in df.columns:
            df.rename(columns={'close': 'price'}, inplace=True)

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # Split the data
        chunks = split_dataframe(df, parts=5, overlap=6)

        # Clean filename and write each chunk
        base_name = os.path.basename(file_path).replace('.csv.gz', '').replace('.csv', '')
        for i, chunk in enumerate(chunks):
            out_path = os.path.join(output_dir, f"part{i+1}_{base_name}.csv.gz")
            chunk.to_csv(out_path, compression='gzip')
            print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

