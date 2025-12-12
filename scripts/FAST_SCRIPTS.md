# Fast Execution Scripts

Quick testing scripts for rapid iteration and validation.

## 🚀 Quick Test Script

**File**: `scripts/quick_test.ps1`

**Purpose**: Balanced fast execution with reasonable results

**Configuration**:
- Dataset: ETTh1 (7 variates - smallest)
- Input length: 48 time steps
- Prediction length: 24 time steps
- Model: Small (d_model=64, 1 layer)
- Epochs: 2
- Batch size: 32

**Execution time**: ~2-5 minutes (depending on hardware)

**Run**:
```powershell
.\scripts\quick_test.ps1
```

---

## ⚡ Ultra-Fast Minimal Test

**File**: `scripts/quick_test_minimal.ps1`

**Purpose**: Absolute minimum configuration for fastest possible execution

**Configuration**:
- Dataset: ETTh1 (7 variates)
- Input length: 24 time steps
- Prediction length: 12 time steps
- Model: Tiny (d_model=32, 1 layer, 2 heads)
- Epochs: 1
- Batch size: 64

**Execution time**: ~30 seconds - 2 minutes

**Run**:
```powershell
.\scripts\quick_test_minimal.ps1
```

---

## 📊 Comparison

| Script | Input | Prediction | Model Size | Epochs | Speed | Use Case |
|--------|-------|-----------|------------|--------|-------|----------|
| `quick_test.ps1` | 48 | 24 | Small (64) | 2 | Fast | Quick validation |
| `quick_test_minimal.ps1` | 24 | 12 | Tiny (32) | 1 | Ultra-fast | Smoke test |

---

## 💡 Tips for Faster Execution

1. **Use CPU**: Scripts automatically use CPU if GPU not available
2. **Reduce epochs**: Already set to minimum (1-2 epochs)
3. **Smaller dataset**: ETTh1 is the smallest (7 variates)
4. **Smaller model**: Already using minimal model sizes
5. **Larger batch size**: Increases throughput (already optimized)

---

## 🎯 When to Use

- **quick_test.ps1**: When you want reasonable results quickly
- **quick_test_minimal.ps1**: When you just need to verify the code works

---

## 📝 Note

These scripts are for testing only. For actual experiments, use the full scripts in:
- `multivariate_forecasting/`
- `boost_performance/`
- etc.

