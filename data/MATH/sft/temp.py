from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
def split_jsonl(input_file, chunk_size=300):
    input_path = Path(input_file)

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        output_file = input_path.with_name(
            f"{input_path.stem}_{i // chunk_size + 1}.jsonl"
        )

        with output_file.open("w", encoding="utf-8") as f:
            f.writelines(chunk)

        print(f"Created {output_file} ({len(chunk)} lines)")


if __name__ == "__main__":
    split_jsonl(ROOT / "/data/MATH/sft/sft_train.jsonl", chunk_size=300)