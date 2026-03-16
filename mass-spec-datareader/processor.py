import os
import glob
import subprocess
import shutil
import click
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pyteomics import mzml
from pyimzml.ImzMLParser import ImzMLParser

# Set professional plotting style
sns.set_theme(style="whitegrid", context="talk", palette="viridis")

class ThermoConverter:
    """Handles the background 'cracking' of Thermo .raw files on macOS/Linux."""
    def __init__(self, parser_path):
        self.parser_path = parser_path

    def convert(self, raw_file, temp_dir):
        print(f"--- Cracking Thermo Raw File: {os.path.basename(raw_file)} ---")
        try:
            cmd = [
                self.parser_path,
                "-i", raw_file,
                "-o", temp_dir,
                "-f", "0", # 0 = mzML
                "-p"        # Peak picking (centroiding)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            mzml_file = os.path.join(temp_dir, os.path.basename(raw_file).replace('.raw', '.mzML'))
            return mzml_file
        except Exception as e:
            print(f"Error converting {raw_file}: {e}")
            return None

class MSProcessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_chromatogram_plot(self, df, base_name):
        """Creates publication-quality TIC and BPC chromatograms."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot Total Ion Chromatogram (TIC)
        ax1.plot(df['rt'], df['tic'], color='#2c7bb6', linewidth=1.5)
        ax1.fill_between(df['rt'], df['tic'], alpha=0.2, color='#2c7bb6')
        ax1.set_ylabel("TIC Intensity")
        ax1.set_title(f"Chromatograms: {base_name}")
        
        # Plot Base Peak Chromatogram (BPC)
        ax2.plot(df['rt'], df['base_peak_int'], color='#d7191c', linewidth=1.5)
        ax2.fill_between(df['rt'], df['base_peak_int'], alpha=0.2, color='#d7191c')
        ax2.set_ylabel("Base Peak Intensity")
        ax2.set_xlabel("Retention Time (min)")
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{base_name}_chromatogram.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Created chromatogram plot: {plot_path}")

    def generate_spatial_heatmap(self, df, base_name):
        """Creates a spatial heatmap of the Total Ion Current (TIC)."""
        # Pivot the coordinates for heatmap plotting
        pivot_df = df.to_pandas().pivot(index='y', columns='x', values='tic')
        
        plt.figure(figsize=(10, 8))
        # Use square root scaling for better visual dynamic range
        sns.heatmap(np.sqrt(pivot_df), cmap="viridis", cbar_kws={'label': 'sqrt(TIC Intensity)'})
        plt.title(f"Spatial TIC Map: {base_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{base_name}_spatial_map.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Created spatial heatmap: {plot_path}")

    def process_mzml(self, file_path, is_temp=False):
        base_name = os.path.basename(file_path).replace('.mzML', '')
        print(f"Parsing Data: {base_name}")
        results = []
        
        with mzml.read(file_path) as reader:
            for spec in reader:
                rt = spec.get('scanList', {}).get('scan', [{}])[0].get('scan start time', 0)
                ms_level = spec.get('ms level', 1)
                mzs = spec.get('m/z array', np.array([]))
                intensities = spec.get('intensity array', np.array([]))
                
                if len(intensities) > 0:
                    results.append({
                        'rt': rt,
                        'ms_level': ms_level,
                        'tic': np.sum(intensities),
                        'base_peak_mz': mzs[np.argmax(intensities)],
                        'base_peak_int': np.max(intensities)
                    })
        
        df = pl.DataFrame(results)
        df.write_parquet(os.path.join(self.output_dir, f"{base_name}_results.parquet"))
        
        # Generate Visualizations
        self.generate_chromatogram_plot(df, base_name)
        
        if is_temp:
            os.remove(file_path)

    def process_imzml(self, file_path):
        base_name = os.path.basename(file_path).replace('.imzML', '')
        print(f"Processing Imaging Data: {base_name}")
        parser = ImzMLParser(file_path)
        
        coords = []
        for idx, (x, y, z) in enumerate(parser.coordinates):
            mzs, intensities = parser.get_spectrum(idx)
            if len(intensities) > 0:
                coords.append({'x': x, 'y': y, 'tic': np.sum(intensities)})
        
        df = pl.DataFrame(coords)
        df.write_csv(os.path.join(self.output_dir, f"{base_name}_spatial_data.csv"))
        
        # Generate Visualizations
        self.generate_spatial_heatmap(df, base_name)

@click.command()
@click.option('--input-dir', '-i', required=True, help='Directory containing .raw, .mzML, or .imzML files')
@click.option('--output-dir', '-o', default='analysis_results', help='Where to save results')
@click.option('--parser-path', default=os.path.join(os.path.dirname(__file__), 'osx-arm64', 'ThermoRawFileParser'), help='Path to the ThermoRawFileParser binary')
def main(input_dir, output_dir, parser_path):
    converter = ThermoConverter(parser_path)
    processor = MSProcessor(output_dir)
    
    # Process .raw files
    raw_files = glob.glob(os.path.join(input_dir, "*.raw"))
    if raw_files:
        temp_dir = os.path.join(output_dir, "temp_mzml")
        os.makedirs(temp_dir, exist_ok=True)
        for raw in raw_files:
            mzml_path = converter.convert(raw, temp_dir)
            if mzml_path:
                processor.process_mzml(mzml_path, is_temp=True)
        shutil.rmtree(temp_dir)

    # Process .mzML files
    for f in glob.glob(os.path.join(input_dir, "*.mzML")):
        processor.process_mzml(f)

    # Process .imzML files
    for f in glob.glob(os.path.join(input_dir, "*.imzML")):
        processor.process_imzml(f)

    print("\n--- Pipeline complete. Visualizations saved in results folder ---")

if __name__ == '__main__':
    main()
