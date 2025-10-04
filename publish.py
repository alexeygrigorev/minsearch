#!/usr/bin/env python
"""
Script to increment version and publish package to PyPI.
Increments the last part of the version (patch version).
"""
import subprocess
import sys
from pathlib import Path


def get_current_version():
    """Read current version from __version__.py"""
    version_file = Path(__file__).parent / "minsearch" / "__version__.py"
    with open(version_file, 'r') as f:
        content = f.read()
    
    # Extract version string
    for line in content.split('\n'):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            return version
    
    raise ValueError("Could not find __version__ in __version__.py")


def increment_version(version):
    """Increment the patch version (last part)"""
    parts = version.split('.')
    parts[-1] = str(int(parts[-1]) + 1)
    return '.'.join(parts)


def update_version(new_version):
    """Update version in __version__.py"""
    version_file = Path(__file__).parent / "minsearch" / "__version__.py"
    with open(version_file, 'w') as f:
        f.write(f"__version__ = '{new_version}'\n")
    print(f"✓ Updated version to {new_version}")


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n→ {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Error: {description} failed")
        print(result.stderr)
        sys.exit(1)
    
    if result.stdout:
        print(result.stdout)
    print(f"✓ {description} completed")


def main():
    print("=" * 60)
    print("Package Publication Script")
    print("=" * 60)
    
    # Get current version
    current_version = get_current_version()
    print(f"\nCurrent version: {current_version}")
    
    # Increment version
    new_version = increment_version(current_version)
    print(f"New version: {new_version}")
    
    # Confirm
    response = input("\nProceed with version increment and publish? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Update version
    update_version(new_version)
    
    # Build package
    run_command("python -m uv run hatch build", "Building package")
    
    # Publish to test PyPI
    response = input("\nPublish to test PyPI? (y/n): ")
    if response.lower() == 'y':
        run_command("python -m uv run hatch publish --repo test", "Publishing to test PyPI")
    
    # Publish to PyPI
    response = input("\nPublish to production PyPI? (y/n): ")
    if response.lower() == 'y':
        run_command("python -m uv run hatch publish", "Publishing to PyPI")
    
    # Clean up
    response = input("\nClean up dist directory? (y/n): ")
    if response.lower() == 'y':
        import shutil
        dist_dir = Path(__file__).parent / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
            print("✓ Cleaned up dist directory")
    
    print("\n" + "=" * 60)
    print(f"Publication complete! Version {new_version}")
    print("=" * 60)


if __name__ == "__main__":
    main()
