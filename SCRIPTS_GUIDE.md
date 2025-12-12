# iTransformer Scripts Guide

A minimal guide to understand what each script does.

## 📁 Script Categories

### 1. **multivariate_forecasting/** - Main Forecasting Experiments
**Purpose**: Standard time series forecasting with different prediction horizons.

**What it does**:
- Trains iTransformer on various datasets
- Tests with 4 different prediction lengths: 96, 192, 336, 720 time steps
- Evaluates model performance on real-world datasets

**Available scripts**:
- `ETT/` - ETT dataset (ETTh1, ETTh2, ETTm1, ETTm2) - 7 variates
- `Traffic/` - Traffic dataset - 862 variates
- `ECL/` - Electricity dataset - 321 variates
- `Weather/` - Weather dataset - 21 variates
- `Exchange/` - Exchange rate dataset - 8 variates
- `PEMS/` - PEMS traffic datasets - Various variates
- `SolarEnergy/` - Solar energy dataset

**Example**: `bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh`

---

### 2. **boost_performance/** - Compare Transformer vs iTransformer
**Purpose**: Demonstrates performance improvement of inverted architecture.

**What it does**:
- Compares vanilla Transformer with iTransformer
- Shows how inverting the architecture improves forecasting
- Tests on ECL, Traffic, and Weather datasets

**Available scripts**:
- `ECL/iTransformer.sh` - Compare on Electricity dataset
- `Traffic/iTransformer.sh` - Compare on Traffic dataset
- `Weather/iTransformer.sh` - Compare on Weather dataset
- Also includes: `iInformer.sh`, `iReformer.sh`, `iFlowformer.sh` variants

**How to use**: Edit script to change `model_name` between `Transformer` and `iTransformer`

**Example**: `bash ./scripts/boost_performance/Weather/iTransformer.sh`

---

### 3. **variate_generalization/** - Zero-Shot Generalization
**Purpose**: Train on partial variates, test on all variates (zero-shot capability).

**What it does**:
- Trains model on only 20% of variates
- Tests on 100% of variates without retraining
- Demonstrates transferable representation learning

**Available scripts**:
- `ECL/iTransformer.sh` - Train on 64/321 variates, test on all 321
- `Traffic/iTransformer.sh` - Train on 172/862 variates, test on all 862
- `SolarEnergy/iTransformer.sh` - Train on partial, test on all
- Also includes: `iInformer.sh`, `iReformer.sh`, `iFlowformer.sh` variants

**Key feature**: Shows iTransformer can generalize to unseen variates

**Example**: `bash ./scripts/variate_generalization/Traffic/iTransformer.sh`

---

### 4. **increasing_lookback/** - Test with Longer Input Sequences
**Purpose**: Evaluates model performance with increasing input sequence lengths.

**What it does**:
- Tests with different lookback windows: 48, 96, 192, 336, 720 time steps
- Shows how iTransformer handles longer input sequences
- Demonstrates scalability advantage

**Available scripts**:
- `ECL/iTransformer.sh` - Test on Electricity with increasing lookback
- `Traffic/iTransformer.sh` - Test on Traffic with increasing lookback
- Also includes: `iInformer.sh`, `iReformer.sh`, `iFlowformer.sh` variants

**Key insight**: iTransformer performs better with longer lookback windows

**Example**: `bash ./scripts/increasing_lookback/Traffic/iTransformer.sh`

---

### 5. **model_efficiency/** - Speed & Memory Optimization
**Purpose**: Tests efficient attention mechanisms for faster training.

**What it does**:
- Uses FlashAttention for hardware acceleration
- Compares training speed and memory usage
- Optimizes for datasets with many variates

**Available scripts**:
- `ECL/iFlashTransformer.sh` - FlashAttention on Electricity
- `Traffic/iFlashTransformer.sh` - FlashAttention on Traffic
- `Weather/iFlashTransformer.sh` - FlashAttention on Weather

**Key benefit**: Faster training with same or better performance

**Example**: `bash ./scripts/model_efficiency/Traffic/iFlashTransformer.sh`

---

## 🔧 Model Variants

Each script category supports multiple model variants:

| Model | Description |
|-------|-------------|
| `iTransformer` | Standard inverted Transformer |
| `iInformer` | iTransformer + ProbSparse attention |
| `iReformer` | iTransformer + Reformer attention |
| `iFlowformer` | iTransformer + Flow attention |
| `iFlashformer` | iTransformer + FlashAttention |

---

## 📊 Quick Reference

| Script Category | Purpose | Key Feature |
|----------------|---------|------------|
| `multivariate_forecasting/` | Main experiments | Standard forecasting |
| `boost_performance/` | Comparison | Shows improvement |
| `variate_generalization/` | Zero-shot | Train on 20%, test on 100% |
| `increasing_lookback/` | Scalability | Longer input sequences |
| `model_efficiency/` | Optimization | Faster training |

---

## 🚀 Quick Start

**For Windows (PowerShell)**:
```powershell
# Main forecasting
.\scripts\multivariate_forecasting\ETT\iTransformer_ETTh1.ps1

# Compare models
.\scripts\boost_performance\Weather\iTransformer.ps1

# Zero-shot generalization
.\scripts\variate_generalization\Traffic\iTransformer.ps1
```

**For Linux/Mac (Bash)**:
```bash
# Main forecasting
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare models
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Zero-shot generalization
bash ./scripts/variate_generalization/ECL/iTransformer.sh
```

---

## 📝 Notes

- All scripts train and test automatically
- Results saved in `./checkpoints/` and `./results/`
- GPU is used automatically if available
- Each script runs multiple experiments (different prediction lengths or lookback windows)

