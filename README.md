# Value Space Codebase

Analysis pipeline for 2-photon calcium imaging and electrophysiology with behavioral data for the value space project. Aligns neural traces to Axon (.abf) stimulus recordings, segments ROIs by cortical region, runs stimulus decoding, and extracts pupil dynamics via DeepLabCut.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate runyan_lab_alignment
```

### 3. Verify installation

```bash
python -c "import numpy, scipy, pandas, matplotlib, sklearn, neo, oasis; print('All packages loaded.')"
```

> **Note:** `deeplabcut` requires a GPU for model training. It is included in the environment for inference and can be skipped if unavailable — install the rest with:
> ```bash
> conda env create -f environment.yml
> # if deeplabcut install fails, remove it from environment.yml and re-run
> ```

## Project structure

```
Calcium imaging/
├── Data alignment/
│   ├── abf_imaging_alignment.py        # Core module: aligns ABF + suite2p, builds trial tensors
│   ├── processing_main_abf.ipynb       # Main notebook: run alignment end-to-end
│   ├── segment_rois_by_region_full.py  # Assign ROIs to cortical regions from border image
│   └── Segment ROIs by Anatomical Region.ipynb
├── Neural data analysis/
│   ├── neural_analysis.py              # Stimulus-response analysis, decoding, regression
│   ├── neural_analysis_main.ipynb      # Population heatmaps, single-neuron selectivity
│   ├── amygdala_analysis.py            # GRIN/amygdala sessions: PSTHs, selectivity (AUC), controls, FOV maps
│   ├── neural_analysis_amygdala.ipynb  # Run the amygdala analysis for one mouse/session
│   ├── decoding_main.ipynb             # Pairwise + temporal stimulus decoding
│   └── kinematics_regression.ipynb     # Ridge regression to remove locomotion signals
├── Pupil alignment/
│   ├── pupil_pipeline.py               # DLC-based pupillometry pipeline
│   └── pupil_main.ipynb                # Run pupil extraction + QC plots
Neuropixel/                             # Open-field Neuropixels: tracking, approach events, session dynamics
```

## Usage

Each sub-pipeline is driven by its corresponding Jupyter notebook. Open the notebooks from the activated environment:

```bash
conda activate runyan_lab_alignment
jupyter notebook
```

**Typical workflow:**

1. **Data alignment** — Run `processing_main_abf.ipynb` to align suite2p traces with ABF stimulus recordings and build trial-locked tensors.
2. **ROI segmentation** — Run `Segment ROIs by Anatomical Region.ipynb` to assign neurons to cortical regions.
3. **Neural analysis** — Run `neural_analysis_main.ipynb` and `decoding_main.ipynb` for population responses and stimulus decoding.
4. **Pupil tracking** — Run `pupil_main.ipynb` for DLC-based pupillometry.

## Notebook outputs

Notebooks are committed without outputs. After cloning, register the filter that
`.gitattributes` refers to (once per clone):

```bash
git config filter.strip-nb-output.clean "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"
git config filter.strip-nb-output.smudge cat
```

## External tools

- **ffmpeg** is required for pupil video processing (installed via conda in the environment).
- **suite2p** outputs (stat.npy, ops.npy, F.npy) are expected as input — run suite2p separately.
