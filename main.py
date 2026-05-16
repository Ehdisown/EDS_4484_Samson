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

    def generate_static_visualizations(self):
        """Module 4: Generates the static engineering plots."""
        try:
            if self.df is None:
                raise ValueError("No data available for visualization.")
            
            # Chart 1: The Wind Power Curve Scatter Plot
            plt.figure(figsize=(8, 5))
            plt.scatter(self.df['generator_speed'], self.df['active_power_calculated_by_converter'], c=self.df['generator_winding_temp_max'], cmap='jet', alpha=0.6, s=10)
            plt.colorbar(label='Generator Winding Temp (°C)')
            plt.xlabel('Generator Rotational Speed (RPM)')
            plt.ylabel('Active Power Output (kW)')
            plt.title('Empirical Power Curve Associated with Thermal Loading')
            plt.grid(True, linestyle='--')
            plt.savefig(os.path.join(self.output_dir, 'static_power_curve.png'), dpi=300)
            plt.close()

            # Chart 2: Distribution Histogram of Thermal Gradients
            plt.figure(figsize=(8, 5))
            plt.hist(self.df['thermal_gradient'], bins=40, color='crimson', edgecolor='black', alpha=0.7)
            plt.xlabel('Thermal Gradient (Generator Temp - Nacelle Ambient Temp) [°C]')
            plt.ylabel('Frequency Count')
            plt.title('Distribution Profile of Internal Nacelle Heat Accumulation')
            plt.savefig(os.path.join(self.output_dir, 'static_thermal_distribution.png'), dpi=300)
            plt.close()

            # Chart 3: Boxplot comparing distinct operational groups
            plt.figure(figsize=(7, 5))
            high_load = self.df[self.df['active_power_calculated_by_converter'] > self.df['active_power_calculated_by_converter'].median()]['generator_winding_temp_max']
            low_load = self.df[self.df['active_power_calculated_by_converter'] <= self.df['active_power_calculated_by_converter'].median()]['generator_winding_temp_max']
            plt.boxplot([low_load, high_load], labels=['Low Power Regime', 'High Power Regime'])
            plt.ylabel('Generator Winding Temperature (°C)')
            plt.title('Thermal stress comparison Across Operational Power Load Profiles')
            plt.savefig(os.path.join(self.output_dir, 'static_load_boxplot.png'), dpi=300)
            plt.close()

            print("[VISUALIZATION SUCCESS] 3 Static engineering charts successfully generated and saved.")
        except Exception as e:
            print(f"[VISUALIZATION ERROR] Failed to output static graphs: {e}")

    def generate_animations(self):
        """Module 5: Generates automated animation files."""
        try:
            if self.df is None:
                raise ValueError("No data available for animation processing.")
            
            print("[INFO] Initiating optimized lightweight animation compilation...")
            
            df_sorted = self.df.reset_index(drop=True).iloc[::100].copy()
            df_sorted["Timeline_Index"] = range(len(df_sorted))
            df_limited = df_sorted.head(100).copy()
            
            # ANIMATION 1: Matplotlib Time-Evolution of Nacelle Temperatures
            fig, ax = plt.subplots(figsize=(8, 4))
            line, = ax.plot([], [], 'r-', label='Nacelle Air Enclosure Temp')
            
            ax.set_xlim(0, len(df_limited))
            ax.set_ylim(df_limited['nacelle_temp'].min() - 2, df_limited['nacelle_temp'].max() + 2)
            ax.set_xlabel('Downsampled Operational Chronological Sequence Index')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title('Real-time Dynamic Ambient Nacelle Thermal Fluctuations')
            ax.grid(True)

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                x_vals = df_limited["Timeline_Index"].iloc[:frame].tolist()
                y_vals = df_limited["nacelle_temp"].iloc[:frame].tolist()
                line.set_data(x_vals, y_vals)
                return line,

            ani = animation.FuncAnimation(
                fig, update, frames=len(df_limited), init_func=init, blit=True, repeat=False
            )
            ani.save(os.path.join(self.output_dir, 'animated_thermal_timeline.gif'), writer='pillow', fps=15)
            plt.close()

            # ANIMATION 2: Plotly Express HTML Interactive Animated Scatter Profile (Fixed Gaps)
            df_limited['Hour_block'] = df_limited["Timeline_Index"].astype(str)
            
            fig_plotly = px.scatter(
                df_limited, 
                x="generator_speed", 
                y="active_power_calculated_by_converter", 
                animation_frame="Hour_block",
                color="generator_winding_temp_max",
                title="Dynamic Kinetic Power Generation Mapping vs Thermal Loads",
                labels={
                    "generator_speed": "Generator Rotational Speed (RPM)", 
                    "active_power_calculated_by_converter": "Active Power (kW)",
                    "Hour_block": "Data Step Segment"
                },
                range_x=[df_limited["generator_speed"].min() * 0.9, df_limited["generator_speed"].max() * 1.1],
                range_y=[df_limited["active_power_calculated_by_converter"].min() * 0.9, df_limited["active_power_calculated_by_converter"].max() * 1.1]
            )
            
            fig_plotly.write_html(os.path.join(self.output_dir, 'animated_power_curve_shift.html'))
            
            print("[ANIMATION SUCCESS] 2 Animation exports successfully compiled and written to storage.")
        except Exception as e:
            print(f"[ANIMATION ERROR] Dynamic matrix rendering crashed: {e}")

# Executable entry point mapping sequence
if __name__ == "__main__":
    # Instantiating pipeline object to process file structures natively
    pipeline = NacelleThermalPipeline(
        input_path="data/wind_data.csv", 
        output_dir="outputs/"
    )
    
    # Sequence Executions
    pipeline.ingest_data()
    pipeline.clean_and_filter_data()
    pipeline.perform_engineering_analytics()
    pipeline.generate_static_visualizations()
    pipeline.generate_animations()