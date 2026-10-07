import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ================= CONFIGURATION =================
DATASET_ROOT = Path("/home/gyanendra/Desktop/ABC/czii_data")
OUTPUT_FILE = Path("czii_dataset_detailed_report.txt")
MAX_TREE_DEPTH = 3
MAX_ITEMS_PER_DIR = 12
# =================================================

def get_size_str(size_bytes):
    """Returns human-readable file size."""
    if size_bytes == 0: return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def generate_tree(dir_path, prefix="", depth=0):
    """Generates a truncated directory tree string."""
    if depth > MAX_TREE_DEPTH:
        return ""
    
    tree_str = ""
    try:
        items = list(dir_path.iterdir())
        items.sort()
        # Prioritize directories for better visualization
        items.sort(key=lambda x: x.is_file())
        
        count = len(items)
        for i, item in enumerate(items):
            if i >= MAX_ITEMS_PER_DIR:
                tree_str += f"{prefix}├── ... ({count - i} more items)\n"
                break
                
            connector = "├── " if i < count - 1 else "└── "
            tree_str += f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"
            
            if item.is_dir():
                extension = "│   " if i < count - 1 else "    "
                tree_str += generate_tree(item, prefix + extension, depth + 1)
    except PermissionError:
        tree_str += f"{prefix}├── [ACCESS DENIED]\n"
    return tree_str

def load_json_safe(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def analyze_dataset():
    print(f"Scanning dataset at: {DATASET_ROOT}...")
    
    # Stats containers
    stats = {
        "extensions": defaultdict(lambda: {"count": 0, "size": 0}),
        "zarr_datasets": defaultdict(lambda: {"count": 0, "size": 0}),
        "files": {
            "zarray": [],
            "zattrs": [],
            "zgroup": [],
            "json": [],
            "csv": [],
            "binary_samples": []
        },
        "total_size": 0,
        "total_files": 0
    }

    if not DATASET_ROOT.exists():
        print("Error: Dataset root not found.")
        return None

    for root, dirs, files in os.walk(DATASET_ROOT):
        root_path = Path(root)
        
        # Check if we are inside a .zarr directory
        # This helps group binary chunks by their parent dataset
        is_inside_zarr = ".zarr" in str(root_path)
        zarr_parent = None
        if is_inside_zarr:
            # Find the path component ending in .zarr
            parts = root_path.parts
            for i, part in enumerate(parts):
                if part.endswith(".zarr"):
                    zarr_parent = Path(*parts[:i+1])
                    break

        for file in files:
            file_path = root_path / file
            try:
                size = file_path.stat().st_size
                stats["total_size"] += size
                stats["total_files"] += 1
                
                # Extension Stats
                ext = file_path.suffix.lower() if file_path.suffix else "(no extension)"
                stats["extensions"][ext]["count"] += 1
                stats["extensions"][ext]["size"] += size

                # Specific File Categorization
                if file == '.zarray':
                    stats["files"]["zarray"].append(file_path)
                elif file == '.zattrs':
                    stats["files"]["zattrs"].append(file_path)
                elif file == '.zgroup':
                    stats["files"]["zgroup"].append(file_path)
                elif ext == '.json':
                    stats["files"]["json"].append(file_path)
                elif ext == '.csv':
                    stats["files"]["csv"].append(file_path)
                elif is_inside_zarr and ext == "(no extension)":
                    # This is likely a data chunk
                    if zarr_parent:
                        key = str(zarr_parent.relative_to(DATASET_ROOT))
                        stats["zarr_datasets"][key]["count"] += 1
                        stats["zarr_datasets"][key]["size"] += size
                        if len(stats["files"]["binary_samples"]) < 5:
                            stats["files"]["binary_samples"].append(file_path)
            except Exception:
                pass

    return stats

def write_report(stats):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write(f"DETAILED DATASET REPORT: CZII Cryo-ET Object Identification\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Root Path: {DATASET_ROOT}\n")
        f.write(f"Total Files Scanned: {stats['total_files']}\n")
        f.write(f"Total Dataset Size: {get_size_str(stats['total_size'])}\n")
        f.write("="*80 + "\n\n")

        # 0. File Type Summary (New Section)
        f.write("## 0. FILE TYPE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Extension':<20} | {'Count':<10} | {'Total Size':<15}\n")
        f.write("-" * 80 + "\n")
        sorted_exts = sorted(stats["extensions"].items(), key=lambda x: x[1]['size'], reverse=True)
        for ext, data in sorted_exts:
            f.write(f"{ext:<20} | {data['count']:<10} | {get_size_str(data['size']):<15}\n")
        f.write("\n")

        # 1. Important Paths
        f.write("## 1. IMPORTANT FILE PATHS\n")
        f.write("-" * 80 + "\n")
        priority_files = ['train.csv', 'test.csv', 'sample_submission.csv', 'train', 'test']
        for p in priority_files:
            path = DATASET_ROOT / p
            if path.exists():
                f.write(f"Found: {path}\n")
        f.write("\n")

        # 2. Directory Structure
        f.write("## 2. DIRECTORY STRUCTURE (Truncated)\n")
        f.write("-" * 80 + "\n")
        if DATASET_ROOT.exists():
            f.write(DATASET_ROOT.name + "/\n")
            f.write(generate_tree(DATASET_ROOT))
        f.write("\n")

        # 3. Zarray Details
        f.write("## 3. ZARRAY METADATA (Dimensions & Compression)\n")
        f.write("-" * 80 + "\n")
        for p in stats["files"]["zarray"]:
            d = load_json_safe(p)
            f.write(f"File: {p.relative_to(DATASET_ROOT)}\n")
            f.write(f"  Shape: {d.get('shape')} | Chunks: {d.get('chunks')} | Type: {d.get('dtype')}\n")
            f.write(f"  Compressor: {d.get('compressor', {}).get('id', 'None')}\n\n")
        if not stats["files"]["zarray"]: f.write("No .zarray files found.\n")
        f.write("\n")

        # 4. Expanded Binary/Data Details (New Logic)
        f.write("## 4. EXPANDED BINARY & DATA STORAGE DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write("This section groups binary chunks by their parent Zarr dataset.\n\n")
        f.write(f"{'Zarr Dataset (Logical Name)':<60} | {'Chunks':<8} | {'Size':<10}\n")
        f.write("-" * 80 + "\n")
        
        sorted_zarr = sorted(stats["zarr_datasets"].items(), key=lambda x: x[1]['size'], reverse=True)
        for name, data in sorted_zarr:
            f.write(f"{name:<60} | {data['count']:<8} | {get_size_str(data['size']):<10}\n")
        
        f.write("-" * 80 + "\n")
        f.write("Sample Binary Chunk Filenames (Raw Data):\n")
        for b in stats["files"]["binary_samples"]:
            f.write(f"  - {b.name} (inside {b.parent.name})\n")
        f.write("\n")

        # 5. Zattrs Details
        f.write("## 5. ZATTRS METADATA (Physical Units & Multiscales)\n")
        f.write("-" * 80 + "\n")
        for p in stats["files"]["zattrs"]:
            d = load_json_safe(p)
            f.write(f"File: {p.relative_to(DATASET_ROOT)}\n")
            # formatting complex dicts
            content = json.dumps(d, indent=2)
            if len(content.splitlines()) > 10:
                f.write(f"  Keys: {list(d.keys())}\n")
                f.write(f"  Content (Preview): {str(d)[:300]}...\n\n")
            else:
                f.write(f"  Content:\n{content}\n\n")
        if not stats["files"]["zattrs"]: f.write("No .zattrs files found.\n")
        f.write("\n")

        # 6. Zgroup Details
        f.write("## 6. ZGROUP DETAILS\n")
        f.write("-" * 80 + "\n")
        for p in stats["files"]["zgroup"]:
            d = load_json_safe(p)
            f.write(f"File: {p.relative_to(DATASET_ROOT)} -> Format: {d.get('zarr_format')}\n")
        f.write("\n")

        # 7. JSON Details
        f.write("## 7. JSON METADATA (Picks/Annotations)\n")
        f.write("-" * 80 + "\n")
        for p in stats["files"]["json"]:
            d = load_json_safe(p)
            f.write(f"File: {p.relative_to(DATASET_ROOT)}\n")
            f.write(f"  Keys: {list(d.keys())}\n\n")
        if not stats["files"]["json"]: f.write("No other .json files found.\n")
        f.write("\n")

        # 10. CSV Details
        f.write("## 10. CSV FILE INSPECTION\n")
        f.write("-" * 80 + "\n")
        for p in stats["files"]["csv"]:
            f.write(f"File: {p.relative_to(DATASET_ROOT)}\n")
            try:
                df = pd.read_csv(p, nrows=3)
                f.write(f"  Dimensions: {len(df.columns)} columns\n")
                f.write(f"  Columns: {list(df.columns)}\n")
                f.write(f"  Sample Row:\n{df.iloc[0].to_string(index=True)}\n")
            except Exception as e:
                f.write(f"  Error reading CSV: {e}\n")
            f.write("\n")

    print(f"Report generation complete: {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    stats = analyze_dataset()
    if stats:
        write_report(stats)