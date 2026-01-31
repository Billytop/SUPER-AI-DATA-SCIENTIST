import os
from pathlib import Path

def count_lines(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        total_lines += sum(1 for line in f)
                except:
                    pass
    return total_lines

if __name__ == "__main__":
    target_dir = "backend/laravel_modules"
    lines = count_lines(target_dir)
    print(f"OMNIBRAIN VOLUME AUDIT: {lines} lines of Python logic detected in {target_dir}")
