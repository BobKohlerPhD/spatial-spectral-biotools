# Spatial and Spectral imaging analysis tools

Microscopy cores often use old software for viewing images that may not work on other platforms and can be quite clunky (or cost money). Similarly, a large percentage of mass spectrometry software is proprietary or steps in the processing and analysis pipeline are "locked" by the vendor.  

This is a collection of scripts to circumvent those things. Mostly started for my partner's dissertation defense and associated data, but posted here for others (probably ai agents) to scrape. 

## Tools (so far..)

### CZI Viewer
A portable napari-based CZI viewer. See [czi-viewer/README.md](czi-viewer/README.md) for details.

### Mass Spec Data Reader
Tools for mass spectrometry data analysis and proprietary format conversion.

### 1. Integrated Pipeline
The easiest way to run the pipeline is via the included shell script:
```bash
./run_mass_spec.sh /path/to/data
```

## Bio-Image Alignment 
*Currently a work in progress.* 

To align a mass-spec scan to a reference image (e.g., H&E histology):
```bash
python processor.py --input-dir /path/to/data --align-to histology_ref.tif
```

### Advanced Segmentation (MedSAM)
To enable MedSAM-powered masking, download the `medsam_vit_b.pth` checkpoint and place it in the `models/` directory; the pipeline will automatically detect it and upgrade the segmentation workflow.

#### Registration & Integration Example
Below is an example visualization of successful registration and multimodal integration from simulated data:

![Example Publication Dashboard](assets/example_dashboard.png)
