#!/usr/bin/env python3
"""
Enhanced DataFrame Processor for Google Hipster Microservices.
Organizes output into service-specific folders for LSTM + Prophet training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def quick_enhance_csv_files():
    """Enhances CSV files and organizes them by microservice name."""
    
    # 1. Setup Directories
    source_dir = Path('training_data')
    output_base_dir = Path('enhanced_training_data')
    output_base_dir.mkdir(exist_ok=True)
    
    # Update glob to match your specific file naming pattern
    csv_files = list(source_dir.glob('*.csv'))
    
    print(f"🔧 Found {len(csv_files)} CSV files to enhance...")
    
    enhanced_count = 0
    
    for csv_file in csv_files:
        try:
            # 2. Extract Microservice Name
            # Pattern: service_action_replicas_X_users_Y.csv -> service
            service_name = csv_file.name.split('_')[0]
            
            # Create subfolder for this specific service
            service_folder = output_base_dir / service_name
            service_folder.mkdir(exist_ok=True)
            
            print(f"Processing [{service_name}]: {csv_file.name}")
            
            # 3. Load and Filter
            df = pd.read_csv(csv_file)
            if len(df) < 5:
                print(f"  ⚠️ Skipping {csv_file.name}: Too few rows")
                continue
            
            # --- FEATURE ENGINEERING BLOCK ---
            
            # Add scaling context
            if 'replica_count' in df.columns:
                df['replica_change'] = df['replica_count'].diff().fillna(0)
                df['replica_scaling_up'] = (df['replica_change'] > 0).astype(int)
            
            if 'load_users' in df.columns:
                df['load_change'] = df['load_users'].diff().fillna(0)
                df['load_increasing'] = (df['load_change'] > 0).astype(int)

            # CPU/Mem Regime Classification
            if 'cpu_cores_value' in df.columns:
                cpu_low = df['cpu_cores_value'].quantile(0.33)
                cpu_high = df['cpu_cores_value'].quantile(0.67)
                df['cpu_regime_encoded'] = 1 # Medium
                df.loc[df['cpu_cores_value'] <= cpu_low, 'cpu_regime_encoded'] = 0
                df.loc[df['cpu_cores_value'] >= cpu_high, 'cpu_regime_encoded'] = 2
            
            # Time-based patterns (Prophet seasonality simulation)
            df['time_idx'] = range(len(df))
            norm_time = df['time_idx'] / (len(df) - 1) if len(df) > 1 else 0
            df['hour_pattern'] = 0.5 + 0.3 * np.sin(2 * np.pi * ((norm_time * 24) % 24 - 8) / 24)
            df['growth_trend'] = 1.0 + 0.3 * norm_time

            # Rolling and Lag features (LSTM)
            if len(df) >= 3:
                for col in ['cpu_cores_value', 'mem_bytes_value']:
                    if col in df.columns:
                        df[f'{col}_roll_mean'] = df[col].rolling(3, min_periods=1).mean()
                        df[f'{col}_lag_1'] = df[col].shift(1).fillna(df[col].iloc[0])
            
            # 4. Clean and Save
            # Note: fillna(method='ffill') is deprecated in newer pandas, using ffill()
            df = df.ffill().bfill().fillna(0)
            
            enhanced_filename = f"{csv_file.stem}_enhanced.csv"
            enhanced_path = service_folder / enhanced_filename
            df.to_csv(enhanced_path, index=False)
            
            enhanced_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed {csv_file.name}: {e}")
    
    print(f"\n🎉 Done! {enhanced_count} files organized into {output_base_dir}/")

if __name__ == "__main__":
    quick_enhance_csv_files()