# 🚀 Installation Guide

Quick guide to get started with Hippocampus Memory System.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Git**: For cloning the repository

## 🔧 Installation Steps

### Method 1: Git Clone (Recommended)

```bash
# Clone the repository
git clone https://github.com/Qquarts/Hippo_memory.git

# Navigate to the directory
cd Hippo_memory

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 run_all_experiments.py --quick
```

### Method 2: Download ZIP

1. Download ZIP from [GitHub Releases](https://github.com/Qquarts/Hippo_memory/releases)
2. Extract the archive
3. Open terminal in extracted directory
4. Run: `pip install -r requirements.txt`

## ✅ Verification

### Quick Test (15 seconds)
```bash
python3 run_all_experiments.py --quick
```

Expected output:
```
🏆 TOTAL: 7/7 experiments passed (100%)
🎉 ALL EXPERIMENTS SUCCESSFUL! 🎉
```

### Run Single Experiment
```bash
cd experiments
python3 hippo_ultimate.py
```

Expected: Visualization saved in current directory

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Install dependencies
```bash
pip install numpy matplotlib
```

### Issue: Import Error in experiments

**Solution**: Run from project root or add to PYTHONPATH
```bash
# From project root
python3 experiments/hippo_ultimate.py

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Permission Denied

**Solution**: Use pip with --user flag
```bash
pip install --user -r requirements.txt
```

## 📦 Optional Dependencies

For advanced features (not required for basic usage):

```bash
# For signal processing
pip install scipy

# For large datasets
pip install h5py

# For development
pip install pytest black flake8
```

## 🎯 Next Steps

1. ✅ Run quick tests
2. 📖 Read [README.md](README.md)
3. 🧪 Explore experiments in `experiments/` folder
4. 🔬 Modify parameters and experiment!

## 💡 Tips

- Use `--quick` mode for fast testing
- Each experiment generates a PNG visualization
- Check `CHANGELOG.md` for version history
- Report issues on [GitHub](https://github.com/Qquarts/Hippo_memory/issues)

---

**Need help?** Open an issue on GitHub!
