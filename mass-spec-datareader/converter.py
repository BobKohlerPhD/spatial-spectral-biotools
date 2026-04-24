import anndata as ad
import numpy as np
import polars as pl
import spatialdata as sd
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Identity

class MSDataConverter:
    """Converts Mass Spec data into scverse-compatible SpatialData objects."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def to_anndata(self, df_spatial, base_name):
        """Standard AnnData conversion (common for scRNA-seq integration)."""
        # We treat each pixel as an 'observation'
        coords = df_spatial.select(['x', 'y']).to_numpy()
        # Intensities go into X (if we had spectral info) or obs (for TIC)
        # For this demo, we'll put TIC into obs
        adata = ad.AnnData(
            X=np.zeros((len(df_spatial), 1)), # Placeholder for spectral matrix
            obs=df_spatial.to_pandas(),
            obsm={"spatial": coords}
        )
        adata.var_names = ["TIC"]
        return adata

    def to_spatialdata(self, df_spatial, image_data, base_name):
        """Creates a modern SpatialData object wrapping rasters and matrices."""
        
        # 1. Create a Raster (The Image)
        # Assuming image_data is a numpy array (TIC heatmap)
        raster_data = Image2DModel.parse(image_data, dims=("c", "y", "x"))
        
        # 2. Create the Table (The metadata per pixel)
        adata = self.to_anndata(df_spatial, base_name)
        # Link table to the image
        adata.obs['region'] = base_name
        adata.obs['instance_id'] = np.arange(len(adata))
        table = TableModel.parse(adata, region=base_name, region_key="region", instance_key="instance_id")
        
        # 3. Assemble SpatialData
        sdata = sd.SpatialData(
            images={f"{base_name}_tic": raster_data},
            tables={"table": table}
        )
        
        return sdata

    def save(self, sdata, base_name):
        path = f"{self.output_dir}/{base_name}.zarr"
        sdata.write(path)
        print(f"Exported to SpatialData (OME-Zarr): {path}")
