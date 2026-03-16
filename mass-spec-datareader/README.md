# Mass Spec Data Reader

Tool for reading and processing mass spectrometry (MS) and associated imaging (MSI) data.

## Current Setup (macOS arm64)
- **Parser:** `ThermoRawFileParser` 
- **Analysis:** `pyteomics`, `pyimzML`, and `polars`.

## Requirement
 **Install .NET 8 Runtime:** [Download here](https://dotnet.microsoft.com/en-us/download/dotnet/8.0). Required for Thermo parser on macOS.


## Usage
The easiest way to run the pipeline is via the included shell script:

```bash
./run_mass_spec.sh /path/to/your/raw/data
```

This will:
1.  Create a virtual environment and install dependencies (python).
2.  Parse all `.raw` files using the embedded engine.
3.  Process all data and output results to `analysis_results/`.

### Manual Usage
If you prefer running the Python script directly:
```bash
source .venv/bin/activate
python processor.py --input-dir /path/to/data --output-dir /path/to/output
```

## Workflow
1.  **Auto-Detection:** The script scans your input folder for `.raw`, `.mzML`, and `.imzML` files.
2.  **Seamless Conversion:** `.raw` files are converted to `mzML` in the background (and then cleaned up).
3.  **Analysis:**
    -   **Spectral:** Extracted into high-performance `Parquet` files with TIC normalization and base peak detection.
    -   **Imaging:** Spatial maps of TIC are generated for any `imzML` data.

