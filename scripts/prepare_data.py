import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import random

from ner_core.core.example import Example
from ner_core.data.loaders.factory import LoaderFactory
from ner_core.data.dictionary_matcher import DictionaryMatcher


def parse_split_ratios(split_str: str) -> Dict[str, float]:
    """
    Parse split ratios from command line argument.
    
    Args:
        split_str: String like "train=80,dev=10,test=10" (percentages)
        
    Returns:
        Dict mapping split names to ratios (0.0-1.0)
        
    Raises:
        ValueError: If percentages don't sum to 100 or format is invalid
        
    Examples:
        >>> parse_split_ratios("train=80,dev=10,test=10")
        {'train': 0.8, 'dev': 0.1, 'test': 0.1}
    """
    splits = {}
    for part in split_str.split(','):
        try:
            name, percent = part.strip().split('=')
            splits[name] = float(percent)
        except ValueError:
            raise ValueError(
                f"Invalid split format: '{part}'. "
                f"Expected 'name=percentage' (e.g., train=80,dev=10,test=10)"
            )
    
    total = sum(splits.values())
    if not (99 <= total <= 101):  # Allow small rounding error
        raise ValueError(
            f"Split percentages sum to {total}, must sum to 100. "
            f"Got: {split_str}"
        )
    
    # Convert percentages to ratios
    return {name: pct / 100.0 for name, pct in splits.items()}


def load_examples(input_path: str, loader) -> List[Example]:
    """
    Load examples from file or directory.
    
    For JSONL files, reads line-by-line.
    For directories, processes all files matching supported extensions.
    
    Args:
        input_path: Path to file or directory
        loader: Loader instance to use for parsing
        
    Returns:
        List of Example objects
        
    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If no valid files found in directory
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    examples = []
    
    if path.is_file():
        print(f"Loading file: {path}")
        content = path.read_text(encoding='utf-8')
        
        # Handle JSONL: read line-by-line
        if path.suffix == '.jsonl':
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                if line.strip():
                    try:
                        example = loader.load(line)
                        examples.append(example)
                    except Exception as e:
                        print(f"WARNING: Failed to parse line {line_num}: {e}", file=sys.stderr)
        else:
            # Single document per file (XML, JSON)
            example = loader.load(content)
            examples.append(example)
            
    elif path.is_dir():
        # Get all files matching supported extensions
        files = sorted(
            list(path.glob('*.xml')) + 
            list(path.glob('*.json')) + 
            list(path.glob('*.jsonl'))
        )
        
        if not files:
            raise ValueError(f"No XML, JSON, or JSONL files found in: {input_path}")
        
        print(f"Loading {len(files)} files from directory: {path}")
        for i, file_path in enumerate(files, 1):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Handle JSONL: read line-by-line
                if file_path.suffix == '.jsonl':
                    for line_num, line in enumerate(content.strip().split('\n'), 1):
                        if line.strip():
                            try:
                                example = loader.load(line)
                                examples.append(example)
                            except Exception as e:
                                print(f"WARNING: {file_path.name} line {line_num}: {e}", file=sys.stderr)
                else:
                    # Single document per file
                    example = loader.load(content)
                    examples.append(example)
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(files)} files...")
                    
            except Exception as e:
                print(f"WARNING: Failed to load {file_path.name}: {e}", file=sys.stderr)
                continue

        print(f"Successfully loaded {len(examples)} examples from {len(files)} files")
    
    else:
        raise ValueError(f"Input path is neither file nor directory: {input_path}")
    
    return examples


def split_dataset(examples: List[Example], split_ratios: Dict[str, float]) -> Dict[str, List[Example]]:
    """
    Split examples into train/dev/test sets.
    
    Args:
        examples: List of Example objects
        split_ratios: Dict mapping split names to ratios (must sum to 1.0)
        
    Returns:
        Dict mapping split names to lists of Examples
    """
    # Shuffle for random splits
    shuffled = examples.copy()
    random.seed(42)  # For reproducibility
    random.shuffle(shuffled)
    
    splits = {}
    start_idx = 0
    
    # Sort by split name for consistent ordering
    sorted_names = sorted(split_ratios.keys())
    
    for i, split_name in enumerate(sorted_names):
        ratio = split_ratios[split_name]
        
        # Calculate split size
        if i == len(sorted_names) - 1:
            # Last split gets remainder to handle rounding
            split_size = len(examples) - start_idx
        else:
            split_size = int(len(examples) * ratio)
        
        splits[split_name] = shuffled[start_idx:start_idx + split_size]
        start_idx += split_size
        
        print(f"  {split_name}: {len(splits[split_name])} examples ({ratio*100:.1f}%)")
    
    return splits

def save_jsonl(examples: List[Example], output_path: str):
    """
    Save examples to JSONL file.
    
    Args:
        examples: List of Example objects
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            json_line = json.dumps(example.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")

def process_directory_preserve_splits(input_dir: Path, output_dir: Path, loader, args):
    """
    Process directory of JSONL files, preserving split structure.
    
    Each input file is processed independently and saved with the same name.
    """
    jsonl_files = sorted(list(input_dir.glob('*.jsonl')))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in directory: {input_dir}")
    
    print(f"\nProcessing {len(jsonl_files)} JSONL files (preserving splits)...")
    
    for jsonl_file in jsonl_files:
        print(f"\n  Processing {jsonl_file.name}...")
        
        # Load examples from this file
        content = jsonl_file.read_text(encoding='utf-8')
        examples = []
        for line_num, line in enumerate(content.strip().split('\n'), 1):
            if line.strip():
                try:
                    example = loader.load(line)
                    examples.append(example)
                except Exception as e:
                    print(f"    WARNING: Line {line_num}: {e}", file=sys.stderr)
        
        print(f"    Loaded {len(examples)} examples")
        
        if args.dictionaries:
            #print("\n=== Step 2: Dictionary matching ===")
            matcher = DictionaryMatcher(args.dictionaries)
            examples = matcher.add_matches_batch(examples)
        
        # Save with same filename
        output_file = output_dir / jsonl_file.name
        save_jsonl(examples, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BioNER datasets in jsonl format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""


    Examples:
    # Convert BioC XML to JSONL
    python prepare_data.py --input annotations.xml --format bioc --output train.jsonl
  
    # Create train/dev/test splits (80/10/10)
    python prepare_data.py --input data.xml --output data/ --split train=80,dev=10,test=10
  
    # Add dictionaries to existing splits (preserves boundaries)
    python prepare_data.py --input splits/ --dictionaries dicts/uniprot.json --output dict_splits/
  
    # Regenerate with different dictionaries
    python prepare_data.py --input splits/ --dictionaries dicts/new.json --output dict_splits/ --overwrite
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input file or directory (BioC XML, JSON, JSONL)'
    )
    
    parser.add_argument(
        '--format',
        choices=['bioc', 'json', 'jsonl'],
        help='Input format (auto-detect from extension if omitted)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSONL file or directory (for splits)'
    )

    parser.add_argument(
        '--dictionaries',
        nargs='+',
        help='Dictionary JSON files for entity matching (space-separated)'
    )
    
    parser.add_argument(
        '--split',
        help='Split percentages (e.g., train=80,dev=10,test=10). Only for non-JSONL inputs.'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing output files/directories'
    )
    
    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)

        # Determine if input is JSONL
        is_jsonl_input = False
        if input_path.is_file() and input_path.suffix == '.jsonl':
            is_jsonl_input = True
        elif input_path.is_dir():
            jsonl_files = list(input_path.glob('*.jsonl'))
            if jsonl_files and not list(input_path.glob('*.xml')) and not list(input_path.glob('*.json')):
                is_jsonl_input = True
        
        # Validation: Cannot split JSONL input
        if is_jsonl_input and args.split:
            parser.error(
                "Cannot use --split with JSONL input (data is already split).\n"
                "To create splits, input raw data first:\n"
                "  python prepare_data.py --input data.xml --output splits/ --split train=80,dev=10,test=10"
            )

        ## Validation: Check overwrite
        if output_path.exists() and not args.overwrite:
            if output_path.is_file():
                parser.error(
                    f"Output file already exists: {args.output}\n"
                    f"Use --overwrite to replace it"
                )
            elif output_path.is_dir():
                existing_files = list(output_path.glob('*.jsonl'))
                if existing_files:
                    parser.error(
                        f"Output directory contains {len(existing_files)} JSONL files.\n"
                        f"Use --overwrite to replace them, or specify a different output directory."
                    )


        

        # Step 1: Load data using appropriate loader
        print("\n=== Step 1: Loading data ===")

        # Handle directory of JSONL files separately (preserve splits)
        if input_path.is_dir() and is_jsonl_input:
            loader = LoaderFactory.create('jsonl')
            output_path.mkdir(parents=True, exist_ok=True)
            process_directory_preserve_splits(input_path, output_path, loader, args)
            print("\n✓ Dataset preparation complete!")
            return

        if args.format:
            loader = LoaderFactory.create(args.format)
        else:
            loader = LoaderFactory.from_file(args.input)

        # Load all examples
        examples = load_examples(args.input, loader)
        print(f"Loaded {len(examples)} examples")
        
        # Step 2: Add dictionary matches (optional)
        if args.dictionaries:
            print("\n=== Step 2: Dictionary matching ===")
            matcher = DictionaryMatcher(args.dictionaries)
            examples = matcher.add_matches_batch(examples)
        
        # Step 3: Split dataset (optional) or save directly
        print("\n=== Step 3: Saving output ===")
        if args.split:
            split_ratios = parse_split_ratios(args.split)
            print(f"Creating splits:")
            splits = split_dataset(examples, split_ratios)
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_examples in splits.items():
                output_file = output_path / f"{split_name}.jsonl"
                save_jsonl(split_examples, output_file)
        else:
            save_jsonl(examples, args.output)
        
        print("\n✓ Dataset preparation complete!")
        
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()