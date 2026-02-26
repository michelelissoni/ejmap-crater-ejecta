"""
Replace folder paths with placeholders in Python files (.py and .ipynb).
Asks for confirmation before making each change.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict


# ==============================================================================
# CONFIGURATION - EDIT THESE PATHS AND PLACEHOLDERS
# ==============================================================================

# Define your path replacements here
# Format: {actual_path: placeholder}
# Paths will be matched as substrings, so longer/more specific paths should come first

PATH_REPLACEMENTS = {
    # Example entries - replace these with your actual paths:

    # Add more paths here as needed
    # Note: Order matters! More specific paths should be listed before general ones
}

FILE_EXTENSIONS = ['.py', '.ipynb','.cfg', '.csv', '.slurm']

# ==============================================================================
# END CONFIGURATION
# ==============================================================================


class PathReplacer:
    def __init__(self, replacements: Dict[str, str]):
        # Sort replacements by path length (longest first) to handle nested paths
        self.replacements = dict(sorted(
            replacements.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ))
        self.changes_made = 0
        self.changes_skipped = 0
    
    def find_python_files(self, root_path: Path, recursive: bool = False) -> List[Path]:
        """Find all .py and .ipynb files in the directory tree."""
        python_files = []
        
        wildcard_str = '**/*' if recursive else '*'
        
        for pattern in [wildcard_str + extension for extension in FILE_EXTENSIONS]:
            python_files.extend(root_path.glob(pattern))
        
        # Sort for consistent ordering
        return sorted(python_files)
    
    def find_paths_in_line(self, line: str) -> List[Tuple[str, str]]:
        """
        Find all matching paths in a line and return (path, placeholder) tuples.
        Returns matches sorted by position in line to handle overlaps correctly.
        """
        matches = []
        
        for path, placeholder in self.replacements.items():
            # Escape special regex characters in the path
            escaped_path = re.escape(path)
            
            # Find all occurrences of this path in the line
            for match in re.finditer(escaped_path, line):
                matches.append((match.start(), match.end(), path, placeholder))
        
        # Sort by start position and length (prefer longer matches)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
        # Remove overlapping matches (keep longer/earlier ones)
        non_overlapping = []
        last_end = -1
        
        for start, end, path, placeholder in matches:
            if start >= last_end:
                non_overlapping.append((path, placeholder))
                last_end = end
        
        return non_overlapping
    
    def replace_in_line(self, line: str, path: str, placeholder: str) -> str:
        """Replace the first occurrence of path with placeholder in line."""
        return line.replace(path, placeholder, 1)
    
    def process_py_file(self, filepath: Path, dry_run: bool = False, dont_ask: bool = False) -> int:
        """Process a .py file and replace paths with placeholders."""
        changes = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return 0
        
        new_lines = []
        file_changed = False
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            
            # Find all paths in this line
            matches = self.find_paths_in_line(line)
            
            if matches:
                # Process each match
                for path, placeholder in matches:
                    if path in line:  # Double-check it's still there
                        # Show the change to the user
                        relative_path = filepath.relative_to(Path.cwd())
                        print(f"\n{'='*70}")
                        print(f"File: {relative_path}")
                        print(f"Line {line_num}:")
                        print(f"  Original: {line.rstrip()}")
                        
                        new_line = self.replace_in_line(line, path, placeholder)
                        print(f"  New:      {new_line.rstrip()}")
                        print(f"\nReplace '{path}' with '{placeholder}'?")
                        
                        if dry_run:
                            response = 'n'
                            print("(Dry run mode - no changes will be made)")
                        elif dont_ask:
                            response = 'a'
                        else:
                            response = input("(y)es / (n)o / (a)ll / (q)uit: ").lower().strip()
                        
                        if response == 'q':
                            print("\nQuitting...")
                            return changes
                        elif response == 'a':
                            # Make this change and all future changes
                            line = new_line
                            changes += 1
                            file_changed = True
                            print("✓ Change made (and will auto-approve remaining changes)")
                            # Set a flag to auto-approve
                            return self.process_py_file_auto(filepath, lines, line_num - 1, new_lines, line)
                        elif response == 'y':
                            line = new_line
                            changes += 1
                            file_changed = True
                            print("✓ Change made")
                        else:
                            print("✗ Change skipped")
                            self.changes_skipped += 1
            
            new_lines.append(line)
        
        # Write back if changes were made
        if file_changed and not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                return 0
        
        return changes
    
    def process_py_file_auto(self, filepath: Path, lines: List[str], 
                            start_idx: int, new_lines: List[str], 
                            current_line: str) -> int:
        """Process remaining lines with auto-approval after user selects 'all'."""
        changes = 1  # Count the change that triggered auto mode
        new_lines.append(current_line)
        
        for line_num, line in enumerate(lines[start_idx + 1:], start_idx + 2):
            matches = self.find_paths_in_line(line)
            
            if matches:
                for path, placeholder in matches:
                    if path in line:
                        line = self.replace_in_line(line, path, placeholder)
                        changes += 1
                        print(f"✓ Auto-approved change in line {line_num}")
            
            new_lines.append(line)
        
        # Write the file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
        
        return changes
    
    def process_ipynb_file(self, filepath: Path, dry_run: bool = False, dont_ask: bool = False) -> int:
        """Process a .ipynb Jupyter notebook file."""
        changes = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return 0
        
        file_changed = False
        
        # Process each cell
        for cell_idx, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') in ['code', 'markdown']:
                source = cell.get('source', [])
                
                # Source can be a string or list of strings
                if isinstance(source, str):
                    source = [source]
                
                new_source = []
                
                for line_num, line in enumerate(source, 1):
                    original_line = line
                    matches = self.find_paths_in_line(line)
                    
                    if matches:
                        for path, placeholder in matches:
                            if path in line:
                                relative_path = filepath.relative_to(Path.cwd())
                                print(f"\n{'='*70}")
                                print(f"File: {relative_path}")
                                print(f"Cell {cell_idx + 1}, Line {line_num}:")
                                print(f"  Original: {line.rstrip()}")
                                
                                new_line = self.replace_in_line(line, path, placeholder)
                                print(f"  New:      {new_line.rstrip()}")
                                print(f"\nReplace '{path}' with '{placeholder}'?")
                                
                                if dry_run:
                                    response = 'n'
                                    print("(Dry run mode - no changes will be made)")
                                elif dont_ask:
                                    response = 'a'
                                else:
                                    response = input("(y)es / (n)o / (a)ll / (q)uit: ").lower().strip()
                                
                                if response == 'q':
                                    print("\nQuitting...")
                                    return changes
                                elif response == 'a':
                                    line = new_line
                                    changes += 1
                                    file_changed = True
                                    print("✓ Change made (auto-approving remaining in this file)")
                                    # Continue with auto-approval for this file
                                    return self.process_ipynb_auto(filepath, notebook, cell_idx, 
                                                                   line_num - 1, new_source, line)
                                elif response == 'y':
                                    line = new_line
                                    changes += 1
                                    file_changed = True
                                    print("✓ Change made")
                                else:
                                    print("✗ Change skipped")
                                    self.changes_skipped += 1
                    
                    new_source.append(line)
                
                cell['source'] = new_source
        
        # Write back if changes were made
        if file_changed and not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=1, ensure_ascii=False)
                    f.write('\n')  # Add final newline
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                return 0
        
        return changes
    
    def process_ipynb_auto(self, filepath: Path, notebook: dict, 
                          start_cell: int, start_line: int, 
                          current_source: List[str], current_line: str) -> int:
        """Process remaining cells with auto-approval."""
        changes = 1
        current_source.append(current_line)
        
        # Finish current cell
        cells = notebook.get('cells', [])
        source = cells[start_cell].get('source', [])
        if isinstance(source, str):
            source = [source]
        
        for line in source[start_line + 1:]:
            matches = self.find_paths_in_line(line)
            if matches:
                for path, placeholder in matches:
                    if path in line:
                        line = self.replace_in_line(line, path, placeholder)
                        changes += 1
            current_source.append(line)
        
        cells[start_cell]['source'] = current_source
        
        # Process remaining cells
        for cell_idx in range(start_cell + 1, len(cells)):
            cell = cells[cell_idx]
            if cell.get('cell_type') in ['code', 'markdown']:
                source = cell.get('source', [])
                if isinstance(source, str):
                    source = [source]
                
                new_source = []
                for line in source:
                    matches = self.find_paths_in_line(line)
                    if matches:
                        for path, placeholder in matches:
                            if path in line:
                                line = self.replace_in_line(line, path, placeholder)
                                changes += 1
                    new_source.append(line)
                
                cell['source'] = new_source
        
        # Write the file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
        
        return changes
    
    def process_all_files(self, root_path: Path, dry_run: bool = False, dont_ask: bool = False, recursive: bool = False):
        """Process all Python files in the directory tree."""
        files = self.find_python_files(root_path, recursive = recursive)
        
        if not files:
            print("No Python files found.")
            return
        
        print(f"Found {len(files)} Python file(s) to process.\n")
        
        if dry_run:
            print("=== DRY RUN MODE ===\n")
        
        for filepath in files:
            relative_path = filepath.relative_to(root_path)
            print(f"\nProcessing: {relative_path}")
            
            if filepath.suffix == '.ipynb':
                changes = self.process_ipynb_file(filepath, dry_run, dont_ask = dont_ask)
            elif filepath.suffix in FILE_EXTENSIONS:
                changes = self.process_py_file(filepath, dry_run, dont_ask = dont_ask)
            else:
                continue
            
            self.changes_made += changes
        
        print(f"\n{'='*70}")
        print(f"Summary:")
        print(f"  Total changes made: {self.changes_made}")
        print(f"  Total changes skipped: {self.changes_skipped}")


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Replace folder paths with placeholders in Python files.',
        epilog='Edit the PATH_REPLACEMENTS dictionary in the script to configure your paths.'
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
    parser.add_argument(
        '--dont-ask',
        action='store_true',
        help='Do not ask for authorization before making changes'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Inspect sub-directories as well.'
    )
    parser.add_argument(
        '--replace', 
        nargs=2, 
        required=False,
        help='--replace [STRING1] [STRING2]. Replace STRING1 with STRING2.')

    
    args = parser.parse_args()
    
    # Validate path
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"Error: Path '{args.path}' does not exist.")
        return 1
    
    if not root_path.is_dir():
        print(f"Error: Path '{args.path}' is not a directory.")
        return 1
    
    if args.replace is not None:
        path_replacements = {args.replace[0]: args.replace[1]}
    else:
        # Check if replacements are configured
        if not PATH_REPLACEMENTS or all(k.startswith('/home/username') or 
                                         k.startswith('/Users/username') or
                                         k.startswith('C:\\Users\\username') 
                                         for k in PATH_REPLACEMENTS.keys()):
            print("WARNING: PATH_REPLACEMENTS appears to contain only example paths.")
            print("Please edit the script and configure your actual paths.")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != 'y':
                print("Exiting.")
                return 0
        path_replacements = PATH_REPLACEMENTS
    
    print(f"Searching for Python files in: {root_path}\n")
    print("Configured path replacements:")
    for path, placeholder in path_replacements.items():
        print(f"  {path} → {placeholder}")
    print()
    
    replacer = PathReplacer(path_replacements)
    replacer.process_all_files(root_path, dry_run=args.dry_run, dont_ask=args.dont_ask, recursive=args.recursive)
    
    return 0


if __name__ == '__main__':
    exit(main())
