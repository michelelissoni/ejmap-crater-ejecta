"""
Find all empty directories in a directory tree and add .gitkeep files to them.
This allows Git to track empty directory structures.
"""

import os
import sys
from pathlib import Path


def find_empty_directories(root_path):
    """
    Find all empty directories in the given directory tree.
    
    Args:
        root_path: The root directory to search from
        
    Returns:
        A list of Path objects representing empty directories
    """
    empty_dirs = []
    root = Path(root_path).resolve()
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root):
        current_dir = Path(dirpath)
        
        # Skip .git directories
        if '.git' in current_dir.parts:
            continue
            
        # Check if directory is empty (no files and no subdirectories)
        # We need to check the actual filesystem, not just what os.walk reports
        try:
            contents = list(current_dir.iterdir())
            # Filter out hidden files if you want truly empty dirs
            # visible_contents = [c for c in contents if not c.name.startswith('.')]
            
            if len(contents) == 0:
                empty_dirs.append(current_dir)
        except PermissionError:
            print(f"Warning: Permission denied for {current_dir}", file=sys.stderr)
            continue
    
    return empty_dirs


def add_gitkeep_files(root_path, dry_run=False):
    """
    Add .gitkeep files to all empty directories.
    
    Args:
        root_path: The root directory to search from
        dry_run: If True, only print what would be done without making changes
        
    Returns:
        Number of .gitkeep files created
    """
    empty_dirs = find_empty_directories(root_path)
    count = 0
    
    if not empty_dirs:
        print("No empty directories found.")
        return 0
    
    print(f"Found {len(empty_dirs)} empty director{'y' if len(empty_dirs) == 1 else 'ies'}:\n")
    
    for empty_dir in empty_dirs:
        gitkeep_path = empty_dir / '.gitkeep'
        relative_path = empty_dir.relative_to(Path(root_path).resolve())
        
        if dry_run:
            print(f"Would create: {relative_path / '.gitkeep'}")
        else:
            try:
                # Create an empty .gitkeep file
                gitkeep_path.touch()
                print(f"Created: {relative_path / '.gitkeep'}")
                count += 1
            except Exception as e:
                print(f"Error creating {relative_path / '.gitkeep'}: {e}", file=sys.stderr)
    
    if not dry_run:
        print(f"\nSuccessfully created {count} .gitkeep file{'s' if count != 1 else ''}.")
    
    return count


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add .gitkeep files to empty directories to preserve them in Git.',
        epilog='Example: python add_gitkeep.py /path/to/repo'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root directory to search (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Validate path
    root_path = Path(args.path)
    if not root_path.exists():
        print(f"Error: Path '{args.path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not root_path.is_dir():
        print(f"Error: Path '{args.path}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Searching for empty directories in: {root_path.resolve()}\n")
    
    if args.dry_run:
        print("=== DRY RUN MODE ===\n")
    
    add_gitkeep_files(root_path, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
