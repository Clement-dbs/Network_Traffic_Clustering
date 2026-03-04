#!/usr/bin/env python
import sys
from pathlib import Path
import subprocess

# Add apps/src to Python path for imports
project_root = Path(__file__).resolve().parent
apps_src = project_root / "apps" / "src"
sys.path.insert(0, str(apps_src))


def run_kmeans():
    """Run K-Means clustering analysis."""
    print("\n" + "="*80)
    print("EXECUTING K-MEANS CLUSTERING")
    print("="*80 + "\n")
    try:
        result = subprocess.run([sys.executable, str(apps_src / "k_means.py")], check=True)
        print("\n✓ K-Means clustering completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during K-Means execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_dbscan():
    """Run DBSCAN clustering analysis."""
    print("\n" + "="*80)
    print("EXECUTING DBSCAN CLUSTERING")
    print("="*80 + "\n")
    try:
        result = subprocess.run([sys.executable, str(apps_src / "dbscan.py")], check=True)
        print("\n✓ DBSCAN clustering completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during DBSCAN execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_agglomerative():
    """Run Agglomerative clustering analysis."""
    print("\n" + "="*80)
    print("EXECUTING AGGLOMERATIVE CLUSTERING")
    print("="*80 + "\n")
    try:
        result = subprocess.run([sys.executable, str(apps_src / "agglomerative.py")], check=True)
        print("\n✓ Agglomerative clustering completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error during Agglomerative execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to execute all clustering algorithms."""
    print("\n" + "="*80)
    print("NETWORK TRAFFIC CLUSTERING - MAIN EXECUTION")
    print("="*80)
    print(f"Project root: {project_root}")
    print(f"Python path includes: {apps_src}")
    
    results = {
        "K-Means": run_kmeans(),
        "DBSCAN": run_dbscan(),
        "Agglomerative": run_agglomerative(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    for algo, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{algo:20} : {status}")
    
    print("\n" + "="*80)
    print("All output files have been saved to:")
    print(f"  - Results: {project_root / 'apps' / 'output' / 'results'}/")
    print(f"  - Figures: {project_root / 'apps' / 'output' / 'figures'}/")
    print("="*80 + "\n")
    
    # Return exit code
    failed = sum(1 for success in results.values() if not success)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
