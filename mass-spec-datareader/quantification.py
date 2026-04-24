import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ChemicalROIManager:
    """Handles ROI extraction and statistical comparison for Chemical biology."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def compare_niches(self, df_spatial, mask_dict):
        """
        Compare chemical signatures between different biological niches.
        mask_dict: {'Tumor': binary_mask, 'Stroma': binary_mask}
        """
        results = []
        for niche_name, mask in mask_dict.items():
            # Filter spatial data by mask
            # Assuming mask is 2D array and df_spatial has x, y coords
            # We map pixel coords to mask indices
            h, w = mask.shape
            
            # This is a vectorized way to check which spatial points are in which mask
            # We filter the dataframe where mask[y, x] is True
            df_niche = df_spatial.filter(
                pl.struct(["x", "y"]).map_elements(
                    lambda p: mask[min(p["y"], h-1), min(p["x"], w-1)] > 0,
                    return_dtype=pl.Boolean
                )
            )
            
            if len(df_niche) > 0:
                mean_tic = df_niche['tic'].mean()
                results.append({
                    'Niche': niche_name,
                    'Mean_Intensity': mean_tic,
                    'Voxel_Count': len(df_niche)
                })
        
        return pl.DataFrame(results)

    def plot_niche_comparison(self, stats_df, base_name):
        """Generates a professional bar/violin plot for the ROI comparison."""
        plt.figure(figsize=(8, 6))
        sns.barplot(data=stats_df.to_pandas(), x='Niche', y='Mean_Intensity', palette='mako')
        plt.title(f"Chemical Distribution by Niche: {base_name}", fontweight='bold')
        plt.ylabel("Mean TIC Intensity (a.u.)")
        sns.despine()
        
        plot_path = f"{self.output_dir}/{base_name}_niche_stats.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path

class MetabolicNetwork:
    """Maps co-occurrence networks of chemical species across the tissue."""
    
    def calculate_cooccurrence(self, adata):
        """
        Calculates spatial correlation between different molecular species.
        Analogous to Functional Connectivity in MRI.
        """
        # If adata.X has multiple columns (e.g. different glycans/lipids)
        # We calculate the correlation matrix
        if adata.n_vars < 2:
            return None
            
        corr_matrix = np.corrcoef(adata.X.T)
        return corr_matrix

    def plot_network(self, corr_matrix, var_names, base_name, output_dir):
        """Plots the metabolic connectivity heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, xticklabels=var_names, yticklabels=var_names, 
                    cmap='RdBu_r', center=0, annot=True)
        plt.title(f"Metabolic Connectivity: {base_name}", fontweight='bold')
        
        path = f"{output_dir}/{base_name}_metabolic_network.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

