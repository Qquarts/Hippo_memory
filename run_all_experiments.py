#!/usr/bin/env python3
"""
================================================================================
Hippocampus Memory System - Experiment Runner
================================================================================

Run all experiments sequentially and generate reports.

Usage:
    python run_all_experiments.py [--quick]

Options:
    --quick     Run quick tests only (skip long experiments)

================================================================================
"""

import sys
import os
import subprocess
import time

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

def run_experiment(name, filepath, quick_mode=False):
    """Run a single experiment"""
    print("\n" + "="*70)
    print(f"🧪 RUNNING: {name}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python3', filepath],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({elapsed:.1f}s)")
            return True
        else:
            print(f"❌ FAILED ({elapsed:.1f}s)")
            print("Error:", result.stderr[:500])
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    quick_mode = '--quick' in sys.argv
    
    print("\n" + "="*70)
    print("🧠 HIPPOCAMPUS MEMORY SYSTEM - EXPERIMENT SUITE")
    print("="*70)
    
    if quick_mode:
        print("\n⚡ Quick Mode: Running essential experiments only\n")
    else:
        print("\n🔬 Full Mode: Running all experiments\n")
    
    experiments_dir = os.path.join(os.path.dirname(__file__), 'experiments')
    
    # Define experiments
    experiments = [
        ("1. Ultimate System", "hippo_ultimate.py", False),
        ("2. Sequence Memory", "hippo_seq.py", False),
        ("3. Multi-Sequence (Fast)", "hippo_seq_v2_fast.py", False),
        ("4. Long Sequence (Fast)", "hippo_seq_v3_fast.py", True),
        ("5. Alphabet Memory", "hippo_alphabet.py", True),
        ("6. Word Memory", "hippo_words.py", False),
        ("7. Decision Making", "hippo_branching.py", True),
        ("8. Parallel Branching", "hippo_branching_v2.py", False),
        ("9. Sleep Consolidation", "hippo_dream_final.py", True),
        ("10. CA1 Temporal", "hippo_ca1_temporal.py", True),
        ("11. CA1 Novelty", "hippo_ca1_novelty.py", False),
        ("12. Subiculum Gate", "hippo_subiculum_gate.py", False),
    ]
    
    results = []
    
    for name, filename, skip_in_quick in experiments:
        if quick_mode and skip_in_quick:
            print(f"\n⏩ SKIPPING: {name} (Quick mode)")
            continue
        
        filepath = os.path.join(experiments_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n⚠️  WARNING: {name} - File not found: {filename}")
            results.append((name, False))
            continue
        
        success = run_experiment(name, filepath, quick_mode)
        results.append((name, success))
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXPERIMENT SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "="*70)
    print(f"🏆 TOTAL: {passed}/{total} experiments passed ({100*passed//total}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL EXPERIMENTS SUCCESSFUL! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} experiment(s) failed\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

