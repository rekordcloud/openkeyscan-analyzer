#!/usr/bin/env python3
"""
Script to replace symlinks with actual files in the PyInstaller dist folder.
This ensures the distribution works correctly when zipped or copied to other systems.
"""

import os
import shutil
from pathlib import Path


def dereference_symlinks(dist_path):
    """
    Find all symlinks in the dist folder and replace them with actual file copies.

    Args:
        dist_path (Path): Path to the distribution folder
    """
    dist_path = Path(dist_path)

    if not dist_path.exists():
        print(f"Error: Distribution path {dist_path} does not exist")
        return

    symlinks_found = []

    # Find all symlinks
    for root, dirs, files in os.walk(dist_path):
        root_path = Path(root)
        for name in files + dirs:
            item_path = root_path / name
            if item_path.is_symlink():
                symlinks_found.append(item_path)

    if not symlinks_found:
        print("No symlinks found in distribution folder")
        return

    print(f"Found {len(symlinks_found)} symlinks to dereference:")

    # Replace each symlink with actual file/directory
    for symlink_path in symlinks_found:
        try:
            # Get the target of the symlink
            target = symlink_path.resolve()

            if not target.exists():
                print(f"  ⚠️  Warning: Symlink target does not exist: {symlink_path} -> {target}")
                continue

            # Remove the symlink
            symlink_path.unlink()

            # Copy the actual file/directory
            if target.is_dir():
                shutil.copytree(target, symlink_path)
                print(f"  ✓ Copied directory: {symlink_path.name}")
            else:
                shutil.copy2(target, symlink_path)
                print(f"  ✓ Copied file: {symlink_path.name}")

        except Exception as e:
            print(f"  ✗ Error processing {symlink_path}: {e}")

    print(f"\nSuccessfully dereferenced {len(symlinks_found)} symlinks")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        dist_folder = sys.argv[1]
    else:
        dist_folder = "dist/predict_keys"

    print(f"Dereferencing symlinks in: {dist_folder}")
    print("=" * 70)
    dereference_symlinks(dist_folder)
