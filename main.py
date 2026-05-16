import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go

class NacelleThermalPipeline:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None

    def ingest_data(self):
        """Module 1: Load data with robust error handling."""
        try:
            self.df = pd.read_csv(self.input_path)
            print("Successfully loaded {len(self.df)} rows.")
        except FileNotFoundError:
            print(f"Error: The file at {self.input_path} was not found.")
        except Exception as e:
            print(f"Unexpected ingestion error: {e}")

    def clean_and_filter_data(self):
        """Module 2: Automated Cleaning and Unique Programmatic Filtering."""
        try:
            if self.df is None:
                raise ValueError("No data available to clean. Run ingest_data() first.")

            # 1. Automated standard cleaning steps
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            
            # Identify core engineering channels and drop missing data
            core_channels = ['nacelle_temp', 'generator_winding_temp_max', 'active_power_calculated_by_converter', 'generator_speed']
            self.df.dropna(subset=core_channels, inplace=True)
            
            # Data type correction (Ensure numbers are floats)
            for col in core_channels:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df.dropna(subset=core_channels, inplace=True)
            
            # 2. MANDATORY UNIQUE FILTER LOGIC
            # Isolates records where internal nacelle temperature exceeds 20°C
            self.df = self.df[self.df['nacelle_temp'] > 20.0]
            
            cleaned_count = len(self.df)
            print(f"[CLEANING SUCCESS] Deduplicated and applied unique filter (nacelle_temp > 20). Rows: {initial_count} -> {cleaned_count}")
            
            # Save cleaned dataset back to the project directory structure
            cleaned_path = os.path.join(self.output_dir, "../data/dataset_cleaned.csv")
            self.df.to_csv(cleaned_path, index=False)
            print(f"[DATA EXPORT] Cleaned dataset written to {cleaned_path}")
            
        except Exception as e:
            print(f"[CLEANING ERROR] Pipeline failed during data processing: {e}")
        
    def perform_engineering_analytics(self):
        """Module 3: Advanced Numerical Analysis using Mandatory NumPy Interfacing."""
        try:
            if self.df is None or len(self.df) == 0:
                raise ValueError("Dataset is empty. Cannot perform analytics.")

            # Extract pure NumPy arrays for computing high-level statistics
            gen_temp = np.array(self.df['generator_winding_temp_max'], dtype=np.float64)
            power_out = np.array(self.df['active_power_calculated_by_converter'], dtype=np.float64)
            nacelle_temp = np.array(self.df['nacelle_temp'], dtype=np.float64)

            # Execution of Section IV NumPy Requirements
            stats = {
                "Mean": np.mean(gen_temp),
                "Median": np.median(gen_temp),
                "Std_Deviation": np.std(gen_temp),
                "Variance": np.var(gen_temp) 
            }
            
            # Print metrics to terminal (Note: You must interpret these numbers in your IEEE paper!)
            print("\n" + "="*40 + "\n--- NUMPY COMPLETED ENGINEERING METRICS ---")
            for metric, val in stats.items():
                print(f"Generator Winding Temp {metric}: {val:.4f}")
            print("="*40 + "\n")
            
            # Pearson Correlation Analysis using NumPy
            correlation_matrix = np.corrcoef(gen_temp, power_out)
            print(f"[CORRELATION] Gen Winding Temp vs. Converter Active Power: {correlation_matrix[0,1]:.4f}\n")
            
            # Feature engineering: Create a thermodynamic gradient column
            self.df['thermal_gradient'] = gen_temp - nacelle_temp
            return stats
            
        except Exception as e:
            print(f"[ANALYTICS ERROR] Mathematical processing crashed: {e}") 