import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = "/Users/jortvanderweijde/Desktop/Pythontest/"
TF1_DIR = "15minute"     
TF2_DIR = "4hour"    
OUTPUT_DIR = f"CombiTimeframe/Tests"

# Create output folder if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, OUTPUT_DIR), exist_ok=True)

def is_valid_file(filename):
    return not filename.startswith(".") and filename.endswith(".csv")

# --- Get matching file pairs ---
def get_matching_files(tf1_dir, tf2_dir):
    tf1_files = glob.glob(os.path.join(BASE_DIR, tf1_dir, "*.csv"))
    tf2_files = glob.glob(os.path.join(BASE_DIR, tf2_dir, "*.csv"))

    def extract_base(filename):
        parts = os.path.basename(filename).replace(".csv", "").split("_")
        if len(parts) >= 4:
            base = "_".join([parts[0], parts[1], parts[3]])
            if len(parts) == 5:
                base += f"_{parts[4]}"
            return base
        else:
            raise ValueError(f"Unexpected filename format: {filename}")

    tf1_dict = {extract_base(f): f for f in tf1_files}
    tf2_dict = {extract_base(f): f for f in tf2_files}

    matches = []
    for base in tf1_dict:
        if base in tf2_dict:
            filename_out = f"{base}_.csv"
            matches.append((tf1_dict[base], tf2_dict[base], filename_out))
    return matches

# --- Check if file is GZIP compressed ---
def is_gzip_compressed(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(2)
    return magic == b'\x1f\x8b'

# --- Confirm signals ---
def confirm_signals(file1, file2, filename):
    print(f"Now processing: {filename}")
    try:
        # Check if both files are actually gzip-compressed
        if not is_gzip_compressed(file1):
            raise ValueError(f"{file1} is not gzip-compressed.")
        if not is_gzip_compressed(file2):
            raise ValueError(f"{file2} is not gzip-compressed.")
        
        # Load both DataFrames
        df1 = pd.read_csv(file1, index_col=0, parse_dates=True, compression='gzip')
        df2 = pd.read_csv(file2, index_col=0, parse_dates=True, compression='gzip')

        # Drop duplicate timestamps if any
        df1 = df1[~df1.index.duplicated(keep='first')]
        df2 = df2[~df2.index.duplicated(keep='first')]

        # Upsample df2 to 1-hour frequency, forward-fill last known signal
        df2_upsampled = df2.resample('15 min').ffill()

        # Inner join on timestamps
        combined = df1.join(df2_upsampled, lsuffix="_tf1", rsuffix="_tf2", how="inner")

        # Prepare final confirmed signal DataFrame
        confirmed_cols = {}
        common_cols = [col for col in df1.columns if col != "price" and col in df2.columns]

        for col in common_cols:
            c1 = f"{col}_tf1"
            c2 = f"{col}_tf2"
            if c1 in combined.columns and c2 in combined.columns:
                s = pd.Series(0, index=combined.index)
                both_buy = (combined[c1] == 1) & (combined[c2] == 1)
                both_sell = (combined[c1] == -1) & (combined[c2] == -1)
                s.loc[both_buy] = 1
                s.loc[both_sell] = -1
                confirmed_cols[col] = s
        confirmed = pd.concat(confirmed_cols, axis=1)

        # Insert price column from df1 right after the timestamp
        if 'price' in df1.columns:
            confirmed.insert(0, 'price', df1['price'])

        # Save confirmed signals
        out_path = os.path.join(BASE_DIR, OUTPUT_DIR, filename + ".gz")
        confirmed.to_csv(out_path, compression='gzip')
        print(f"✅ Processed: {filename}")
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

def safe_confirm_signals(f1_path, f2_path, filename):
    try:
        print(f"Starting file: {filename}")
        result = confirm_signals(f1_path, f2_path, filename)
        return {"status": "OK", "file": filename, "result": result}
    except Exception as e:
        return {"status": "ERROR", "file": filename, "error": str(e)}

#  --- Main parallel processing ---
def main():
    pairs = get_matching_files(TF1_DIR, TF2_DIR)
    results = []

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(safe_confirm_signals, f1, f2, filename): filename
            for f1, f2, filename in pairs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                result = future.result()
                if result["status"] == "OK":
                    print(f"[OK] Finished: {result['file']}")
                    results.append(result["result"])
                else:
                    print(f"[ERROR] {result['file']} crashed: {result['error']}")
            except Exception as e:
                # Catches errors from future.result() itself
                print(f"[FATAL] Future crashed: {e}")

    return results

if __name__ == "__main__":
    main()