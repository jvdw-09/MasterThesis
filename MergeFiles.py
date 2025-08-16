import gzip
import glob
import os

input_files = sorted(glob.glob("/Users/jortvanderweijde/Desktop/Pythontest/Parts"))
output_file = "/Users/jortvanderweijde/Desktop/Pythontest/Parts"

#Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

seen_dates = set()
header_written = False

with gzip.open(output_file, 'wt') as outfile:
    for i, fname in enumerate(input_files):
        with gzip.open(fname, 'rt') as f:
            for j, line in enumerate(f):
                if j == 0:
                    if not header_written:
                        outfile.write(line)
                        header_written = True
                    continue  # Skip headers in subsequent files

                # Extract the date from the line (assumes date is the first column)
                date = line.split(',')[0]
                if date not in seen_dates:
                    outfile.write(line)
                    seen_dates.add(date)