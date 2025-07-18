# PPO DistilBERT Model Evaluation Scripts

This repository contains comprehensive evaluation scripts for your trained PPO DistilBERT model for building energy management using Sinergym.

## 📁 Files

- `evaluate_ppo_distilbert.py` - **Recommended**: Automatic model detection and evaluation
- `evaluate_ppo_distilbert_simple.py` - Manual path specification for evaluation
- `README_evaluation.md` - This documentation file

## 🚀 Quick Start

### Option 1: Automatic Evaluation (Recommended)

```bash
python evaluate_ppo_distilbert.py
```

This script will:
1. Automatically find your training workspace
2. Load the most recent trained model
3. Load the reward configuration
4. Run 12 evaluation episodes (one per month)
5. Generate comprehensive plots and statistics

### Option 2: Manual Path Specification

```bash
python evaluate_ppo_distilbert_simple.py
```

This script will prompt you to enter:
- Path to your trained model
- Path to your training workspace

## 📊 What the Evaluation Provides

### 1. **4 Detailed Plots** (High-resolution PNG files)
- **Indoor Temperature** with comfort zone lines
- **Outdoor Temperature** 
- **Energy Usage** (HVAC electricity consumption)
- **Reward** values over time

Each plot includes:
- Mean values (solid line with markers)
- Standard deviation bands (±SD)
- 95% confidence intervals (95% CI)
- Comfort zone reference lines (for indoor temperature)

### 2. **Comprehensive Statistics**
- Episode-wise summary statistics
- Monthly aggregated statistics
- Comfort zone analysis
- Energy efficiency metrics

### 3. **CSV Data Files**
- `ppo_evaluation_summary.csv` - Episode-level summary
- `ppo_evaluation_stepwise_metrics.csv` - Step-by-step data
- `monthly_statistics.csv` - Monthly aggregated data

## 🎯 Evaluation Metrics

The evaluation runs **12 episodes** (one for each month) and tracks:

### Primary Metrics
- **Indoor Temperature** (°C) - Target: 20-23.5°C (winter), 23-26°C (summer)
- **Energy Usage** (W) - HVAC electricity consumption
- **Reward** - Combined comfort and energy efficiency score
- **Outdoor Temperature** (°C) - Environmental conditions

### Comfort Zone Analysis
- **Winter Comfort**: Percentage of time within 20-23.5°C
- **Summer Comfort**: Percentage of time within 23-26°C

### Statistical Measures
- Mean and standard deviation for all metrics
- 95% confidence intervals
- Monthly trends and patterns

## 🔧 Configuration

The evaluation uses the same configuration as your training:

```python
# Reward parameters
lambda_temperature = 28
lambda_energy = 0.01
energy_weight = 0.4

# Environment settings
env_id = 'Eplus-5zone-hot-continuous-v1'
timesteps_per_hour = 1
runperiod = (1, 1, 1991, 31, 12, 1991)  # Full year simulation
```

## 📈 Understanding the Plots

### Indoor Temperature Plot
- **Green dashed lines**: Winter comfort zone (20-23.5°C)
- **Orange dash-dot lines**: Summer comfort zone (23-26°C)
- **Blue line**: Mean indoor temperature
- **Shaded areas**: Statistical uncertainty (SD and CI)

### Energy Usage Plot
- Shows HVAC electricity consumption
- Lower values indicate better energy efficiency
- Seasonal patterns should be visible

### Reward Plot
- Combined score of comfort and energy efficiency
- Higher values indicate better overall performance
- Should show learning progress over time

## 🛠️ Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ❌ Error loading model: [Errno 2] No such file or directory
   ```
   **Solution**: Ensure your model path is correct and the model files exist.

2. **Environment Setup Issues**
   ```
   ❌ Error: No module named 'sinergym'
   ```
   **Solution**: Install Sinergym: `pip install sinergym`

3. **CUDA/GPU Issues**
   ```
   ❌ Error: CUDA out of memory
   ```
   **Solution**: The script uses CPU by default. If you have GPU issues, ensure CUDA is properly configured.

4. **Observation Space Mismatch**
   ```
   ❌ Error: Observation space mismatch
   ```
   **Solution**: Ensure you're using the same `SentenceObservationWrapper` as in training.

### File Structure Requirements

Your training workspace should contain:
```
workspace_path/
├── ppo_distilbert_YYYYMMDD-HHMMSS/  # Model directory
│   ├── model.zip
│   └── ...
├── reward_YYYYMMDD-HHMMSS.json      # Reward configuration
└── progress.csv                     # Training progress
```

## 📋 Output Files

After evaluation, you'll find these files in `workspace_path/evaluation_YYYYMMDD-HHMMSS/`:

### Plots
- `indoor_temperature_evaluation.png`
- `outdoor_temperature_evaluation.png`
- `energy_usage_evaluation.png`
- `reward_evaluation.png`

### Data Files
- `ppo_evaluation_summary.csv`
- `ppo_evaluation_stepwise_metrics.csv`
- `monthly_statistics.csv`

## 🔍 Interpreting Results

### Good Performance Indicators
- **Indoor Temperature**: Stays within comfort zones most of the time
- **Energy Usage**: Low and stable consumption
- **Reward**: Consistently positive values
- **Comfort**: >80% time in comfort zones

### Areas for Improvement
- **High Energy Usage**: May indicate inefficient control
- **Temperature Violations**: Comfort zone breaches
- **Low Rewards**: Poor balance between comfort and efficiency
- **High Variance**: Inconsistent performance

## 🎯 Customization

### Modifying Evaluation Parameters

Edit the script to change:
- Number of episodes: Modify `num_episodes=12` in `run_evaluation()`
- Evaluation period: Change `runperiod` in `setup_environment()`
- Plot styles: Modify `plot_evaluation_graphs()` function

### Adding New Metrics

To track additional metrics:
1. Add extraction in `run_evaluation()` function
2. Update `create_monthly_stats()` function
3. Add new plot in `plot_evaluation_graphs()` function

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify your model and workspace paths
4. Check that your training configuration matches evaluation

## 🏆 Best Practices

1. **Run Multiple Evaluations**: Evaluate your model multiple times to ensure consistency
2. **Compare Baselines**: Compare against rule-based controllers or other RL algorithms
3. **Monitor Trends**: Look for seasonal patterns and learning progress
4. **Validate Comfort**: Ensure the model maintains occupant comfort
5. **Check Energy Efficiency**: Verify that energy savings don't compromise comfort

---

**Happy Evaluating! 🚀**