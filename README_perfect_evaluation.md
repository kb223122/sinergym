# 🚀 Perfect PPO DistilBERT Evaluation Script

A comprehensive, robust evaluation script that works perfectly for all three BERT variants (fixed, trainable, partially trainable) with proper normalization, correct feature extractors, multiple episodes, and professional statistics.

## 📁 Files

- `perfect_evaluation_script.py` - **Main evaluation script** (command line interface)
- `run_evaluation_example.py` - **Simple example script** (no command line needed)
- `README_perfect_evaluation.md` - This documentation

## ✨ Key Features

### ✅ **Universal Compatibility**
- Works with **Fixed BERT** (frozen layers)
- Works with **Trainable BERT** (fully trainable)
- Works with **Partially Trainable BERT** (some layers frozen)

### ✅ **Perfect Training Match**
- **NormalizeObservation** wrapper (matches training)
- **NormalizeAction** wrapper (matches training)
- **Exact same feature extractor** as training
- **Proper custom objects** for model loading

### ✅ **Robust Statistics**
- **12 episodes** (one per month for full year coverage)
- **Proper confidence intervals** (per-month sample counts)
- **Comprehensive comfort zone analysis**
- **Professional plotting** with high resolution

### ✅ **Professional Output**
- **4 detailed plots** with confidence intervals
- **CSV data files** for further analysis
- **Summary statistics** and comfort analysis
- **Error handling** and validation

## 🚀 Quick Start

### Option 1: Simple Example Script (Recommended)

1. **Edit the configuration** in `run_evaluation_example.py`:
```python
MODEL_DIR = '/path/to/your/model/directory'
MODEL_NAME = 'ppo_distilbert_20250709-025523.zip'
BERT_MODE = 'trainable'  # 'fixed', 'trainable', or 'partial'
NUM_EPISODES = 12
OUTPUT_DIR = 'evaluation_results'
```

2. **Run the evaluation**:
```bash
python run_evaluation_example.py
```

### Option 2: Command Line Interface

```bash
python perfect_evaluation_script.py \
    --model_dir /path/to/your/model/directory \
    --model_name ppo_distilbert_20250709-025523.zip \
    --bert_mode trainable \
    --num_episodes 12 \
    --output_dir evaluation_results
```

## 🎯 BERT Mode Selection

### **Fixed BERT** (`--bert_mode fixed`)
- BERT layers are completely frozen
- Only the linear head is trainable
- Use for pre-trained feature extraction

### **Trainable BERT** (`--bert_mode trainable`)
- All BERT layers are fully trainable
- Complete fine-tuning approach
- Use for domain-specific adaptation

### **Partially Trainable BERT** (`--bert_mode partial`)
- Some BERT layers are frozen (first 4 layers)
- Later layers are trainable
- Balance between pre-trained features and adaptation

## 📊 What You Get

### **4 Professional Plots**
1. **Indoor Temperature** with comfort zone lines
2. **Outdoor Temperature** 
3. **Energy Usage** (HVAC consumption)
4. **Reward** values over time

Each plot includes:
- Mean values with markers
- Standard deviation bands (±SD)
- 95% confidence intervals (95% CI)
- Comfort zone reference lines (indoor temperature)
- Professional styling (300 DPI)

### **Data Files**
- `ppo_bert_evaluation_{bert_mode}.csv` - Step-by-step data
- `monthly_stats_{bert_mode}.csv` - Monthly aggregated statistics

### **Console Output**
- Episode progress
- Summary statistics
- Comfort zone analysis
- Performance metrics

## 🔧 Technical Details

### **Environment Setup**
```python
env = gym.make('Eplus-5zone-hot-continuous-v1', config_params={
    'timesteps_per_hour': 1,
    'runperiod': (1, 1, 1991, 31, 12, 1991),
    'reward': reward_kwargs
})

env = LoggerWrapper(env)
env = CSVLogger(env)
env = NormalizeObservation(env)  # ✅ Matches training
env = NormalizeAction(env)       # ✅ Matches training
env = SentenceObservationWrapper(env)
```

### **Model Loading**
```python
model = PPO.load(
    model_path,
    custom_objects={
        "ActorCriticPolicy": PPOBertPolicy,
        "BaseFeaturesExtractor": DistilBERTExtractor
    }
)
```

### **Feature Extractor Selection**
The script automatically selects the correct feature extractor based on `bert_mode`:
- **Fixed**: BERT in eval mode, parameters frozen
- **Trainable**: BERT in train mode, all parameters trainable
- **Partial**: BERT in train mode, first 4 layers frozen

## 📈 Understanding the Results

### **Good Performance Indicators**
- **Indoor Temperature**: Stays within comfort zones (20-23.5°C winter, 23-26°C summer)
- **Energy Usage**: Low and stable consumption
- **Reward**: Consistently positive values
- **Comfort**: >80% time in comfort zones

### **Statistical Measures**
- **Mean ± SD**: Average performance and variability
- **95% CI**: Statistical confidence in the results
- **Monthly Trends**: Seasonal patterns and performance

## 🛠️ Troubleshooting

### **Common Issues**

1. **Model Loading Error**
   ```
   ❌ Error: KeyError: 'ActorCriticPolicy'
   ```
   **Solution**: Ensure you're using the correct `bert_mode` that matches your training.

2. **Observation Space Mismatch**
   ```
   ❌ Error: Observation space mismatch
   ```
   **Solution**: The script automatically handles this with proper wrappers.

3. **File Not Found**
   ```
   ❌ Error: Model file not found
   ```
   **Solution**: Check your `MODEL_DIR` and `MODEL_NAME` paths.

4. **CUDA/GPU Issues**
   ```
   ❌ Error: CUDA out of memory
   ```
   **Solution**: The script uses CPU by default. Ensure CUDA is properly configured if using GPU.

### **Validation Steps**

1. **Check Model Directory**: Ensure it contains:
   - Model file (`.zip`)
   - Reward configuration (`.json`)
   - Training logs

2. **Verify BERT Mode**: Match the mode used during training:
   - Check your training script for the extractor definition
   - Use the corresponding `bert_mode` in evaluation

3. **Test with Single Episode**: Start with `--num_episodes 1` for quick testing

## 🎯 Example Usage for Different Models

### **Fixed BERT Model**
```bash
python perfect_evaluation_script.py \
    --model_dir /path/to/fixed-bert-model \
    --model_name ppo_distilbert_fixed.zip \
    --bert_mode fixed
```

### **Trainable BERT Model**
```bash
python perfect_evaluation_script.py \
    --model_dir /path/to/trainable-bert-model \
    --model_name ppo_distilbert_trainable.zip \
    --bert_mode trainable
```

### **Partially Trainable BERT Model**
```bash
python perfect_evaluation_script.py \
    --model_dir /path/to/partial-bert-model \
    --model_name ppo_distilbert_partial.zip \
    --bert_mode partial
```

## 📋 Output Structure

```
evaluation_results/
├── ppo_bert_evaluation_trainable.csv
├── monthly_stats_trainable.csv
├── indoor_temperature_trainable.png
├── outdoor_temperature_trainable.png
├── energy_usage_trainable.png
└── reward_trainable.png
```

## 🔍 Advanced Customization

### **Modifying Feature Extractors**
Edit the `get_bert_extractor()` function to match your exact training configuration:
- Change activation functions (ReLU vs GELU)
- Modify layer sizes
- Adjust freezing patterns

### **Custom Plotting**
Modify the `plot_evaluation_graphs()` function to:
- Change colors and styles
- Add custom metrics
- Modify plot layouts

### **Additional Metrics**
Add new metrics in the `run_evaluation()` function:
```python
episode_results.append({
    'Episode': episode + 1,
    'Step': step,
    'Month': current_raw_obs[0],
    'Day': current_raw_obs[1],
    'Hour': current_raw_obs[2],
    'Indoor Temperature': current_raw_obs[11],
    'Outdoor Temperature': current_raw_obs[3],
    'Energy Usage': current_raw_obs[15],
    'Reward': reward,
    'Custom Metric': custom_value  # Add your metric here
})
```

## 🏆 Best Practices

1. **Always use the same BERT mode** as in training
2. **Run multiple evaluations** for consistency checking
3. **Compare against baselines** (rule-based controllers)
4. **Monitor comfort zones** and energy efficiency
5. **Validate results** across different seasons

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your model paths and BERT mode
3. Ensure all dependencies are installed
4. Test with a single episode first

---

**Happy Evaluating! 🚀**

This script provides a production-ready evaluation system for your PPO DistilBERT models with professional statistics, beautiful plots, and robust error handling.