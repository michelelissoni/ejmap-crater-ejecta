# Deep learning map of fresh crater ejecta on Mercury
A repository for EJMAP, a deep learning algorithm to map crater ejecta on the surface of Mercury and link them to their crater of origin.

<img src="https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/final_map.png" width="950">

EJMAP consists in two neural network models: EJCONN, a classification model that selects the tiles of a Mercury mosaic where the ejecta from a given crater are located, and EJSEG, a segmentation model which maps the ejecta on those tiles ([explanatory diagram](https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/mapping_procedure.png)).

## Instructions for replication

1. Download or clone this repository. <br/><br/>
2. Go to the [data repository](https://doi.org/10.5281/zenodo.18787894) and download the `results.zip` and `data.zip` files. Unzip them in the repository root folder. There should now be five subfolders:   
      ├── code  
      ├── data  
      ├── images  
      ├── log  
      └── results<br/><br/> 
3. Run the `install.sh` script to adapt the file paths to your directory structure.
   - Alternatively, you can choose to move some subfolders elsewhere (for example, if your computing cluster has multiple storage spaces). In that case, update the paths manually in the config files (`code/*.cfg`) and in the Jupyter Notebooks (`code/*.ipynb`).<br/><br/>
4. Run steps 1, 4, 7 of the Jupyter Notebook `code/ejecta_map_percrater.ipynb` to generate the remaining data files (this could take several hours).
   - Steps 2 and 3 need to be run if you want to modify the manual ejecta masks (in `data/cyl_data`) used for training. You can do so in QGIS with [QClassiPy](https://plugins.qgis.org/plugins/QClassiPy/).
   - Step 5 is used to generate new TV-Test splits.
   - Step 6 generates different mapping (eval) tiles.<br/><br/>
5. Choose the EJCONN and EJSEG versions you want to use. The hyperparameters of the versions are shown in `code/connection_version_hparams.csv` and `code/ejecta_version_hparams.csv`. You can try new hyperparameter combinations by adding rows to these files (each with a unique version number).<br/><br/>
6. Configure your training run by editing `code/train_run_config.cfg`. Launch the training with `launch_training.sh` (local machine) or `training_slurm_launch.sh` (on a computing cluster).
   - If working on a computing cluster, make sure the `#SBATCH` options in the `code/*.slurm` files are appropriate for your cluster. <br/><br/>
7. Check the results in the `results/conn_results/version_{VERSION NUMBER}` and `results/ejc_results/version_{VERSION NUMBER}` folders. The `eval_metrics.csv` files show the evaluation metrics. The `.ckpt` files contain the model weights.
   - For full replication, the `.ckpt` files of each iteration should be retrieved. In `code/train_run_config.cfg`, set `RM_LOG=0`, then copy them from the `version_{VERSION NUMBER}/checkpoints` directory in `log/conn_log` or `log/ejc_log`. Due to their size, this is feasible only for important runs.<br/><br/>
8. Run the Python script `code/create_map.py` (or launch parallel mapping processes with `mapping_launch_slurm.sh`, after configuring `code/mapping_run_config.cfg`) to generate the individual ejecta masks and the color composite map.<br/><br/>
9. Run post-processing operations and analyses using the `code/ejecta_map_percrater_postprocess.ipynb` and `code/crater_pattern_analysis.ipynb` Jupyter Notebooks. 

## File tree

### User-facing files
These scripts and notebooks need to be run to execute the installation, preprocessing, model training and postprocessing. Also included are configuration files that must be edited by the user.

**Installation**  
├── *install.sh* : installer, writes the correct paths for your system  
├── *uninstall.sh* : uninstaller, replaces the root path for your system with a placeholder  

**Preprocessing**  
├── code  
│   ├── *ejecta_map_percrater.ipynb* : this Jupyter Notebook creates all the preprocessed data in 7 steps  

**Training**  
├── code  
│   ├── *train_run_config.cfg* : configure the training runs, edit before running  
│   ├── *connection_version_hparams.csv* : EJCONN hyperparameter configurations, choose those you want to train  
│   ├── *ejecta_version_hparams.csv* : EJSEG hyperparameter configurations, choose those you want to train  
│   ├── *launch_training.sh* : launch the training runs on the local machine, executed sequentially  
│   ├── *training_slurm_launch.sh* : launch training run jobs on the nodes of a cluster  

**Mapping**  
├── code  
│   ├── *create_map.py* : script that generates the individual ejecta masks and the color composite  
│   ├── *mapping_run_config.cfg* : mapping run configuration, edit this to run multiple mappings in parallel  
│   ├── *mapping_slurm_launch.sh* : launch multiple mapping jobs on the nodes of a cluster 

**Postprocessing and analysis**  
├── code  
│   ├── *ejecta_map_percrater_postprocess.ipynb* : code for various post-processing operations  
│   ├── *crater_pattern_analysis.ipynb* : code for crater pattern analysis and comparison with manual maps  

<br/>

### Back-end files
Python scripts called by the bash executables and custom Python modules. The user does not need to interact with these, but they must be edited to modify the algorithm.

**EJCONN: the Connection model**  
├── code  
│   ├── *connection_prepare_versions.py* : configure the training runs  
│   ├── *connection_train.py* : train the model  
│   ├── *connection_test_summarise.py* : gather the training results in the `conn_results` folder  
│   ├── *connection_evaluate_map.py* : evaluate the trained model  
│   ├── *connection_data_module.py* : data module, retrieves the data for training, testing and prediction  
│   ├── *connection_data_preparation.py* : functions handling data preprocessing, model configuration, training and prediction  

**EJSEG: the Segmentation model**  
├── code  
│   ├── *ejecta_prepare_versions.py* : configure the training runs  
│   ├── *ejecta_train.py* : train and evaluate the model  
│   ├── *ejecta_test_summarise.py* : gather the training results in the `ejc_results` folder  
│   ├── *ejecta_data_module.py* : data module, retrieves the data for training, testing and prediction  
│   ├── *ejecta_data_preparation.py* : functions handling data preprocessing, model configuration, training, prediction, mapping  
│   ├── *smp_custom_models.py* : segmentation neural network architecture  

**Planet characteristics**  
├── code  
│   ├── *planet_constants.py* : holds the parameters of the planet. Change to apply to a different planet.

**Utilities**  
├── code  
│   ├── *az_processing.py* : functions handling azimuthal reprojection  
│   ├── *pos_processing.py* : miscellaneous functions for preparing tile positions and handling the data in cylindrical projection  
│   ├── *terratorch_custom_tasks.py* : TerratorchTask, puts together the neural network model and the hyperparameters  
│   ├── *replace_paths.py* : called by the installer and uninstaller script to adapt code to the user's directory structure  
│   ├── *colorlist.txt* : list of colors for plots

**Processing cluster jobs**  
├── code  
│   ├── *training_job.slurm* : training  
│   ├── *summarise_job.slurm* : gather results after training  
│   ├── *evaluate_start.slurm* : triggers EJCONN evaluation  
│   ├── *evaluate_job.slurm* : EJCONN evaluation  
│   ├── *mapping_job.slurm* : mapping  

<br/>

### Data

**Craters**  
├── data  
│   ├── ejc_data  
│   │   ├── *fresh_craters.shp* : the shapefile of the crater rims, also shows the identifying codes (crater FIDs), names, parameters.    

**Ground truth**  
├── data  

│   ├── cyl_data  
│   │   ├── *mask_crater\_{CRATER FID}.tif* : manually-mapped ejecta inside the TVT tiles for a given crater, cylindrical projection.  
│   │   ├── *positions_crater\_{CRATER FID}.csv* : tile positions in cylindrical projection. Access with [QClassiPy](https://plugins.qgis.org/plugins/QClassiPy/) to modify the masks.  

│   ├── ejc_data  
│   │   ├── *human_mask.tif* : manually-mapped ejecta inside the TVT tiles for all craters, cylindrical projection.  
│   │   ├── *human_positions.csv* : tile positions in cylindrical projection. Define additional tiles here.

**TV-Test splits**  
├── data  
│   ├── conn_data  
│   │   ├── *az_positions_all.csv* : list of all TVT azimuthal tiles, the `test_{SPLIT CODE}` columns assign them to TV (0) or Test (1).  

**EJCONN: the Connection model**  
├── data  
│   ├── conn_data  

│   │   ├── images  
│   │   │   ├── *image\_{POS INDEX}\_crater\_{CRATER FID}.npy* : image channels, TVT  

│   │   ├── attention  
│   │   │   ├── *att\_{POS INDEX}\_crater\_{CRATER FID}.npy* : attention channel, TVT  

│   │   ├── eval_positions  
│   │   │   ├── *eval_positions_crater\_{CRATER FID}.csv* : list of azimuthal tile "eval" positions used for the mapping  

│   │   ├── eval_images  
│   │   │   ├── *eval_image\_{EVAL POS INDEX}\_crater\_{CRATER FID}.npy* : image channels, mapping  

│   │   ├── eval_attention  
│   │   │   ├── *att\_{EVAL POS INDEX}\_crater\_{CRATER FID}.npy* : attention channel, mapping  

**EJSEG: the Connection model**  
├── data  
│   ├── ejc_data  

│   │   ├── positions  
│   │   │   ├── *az_positions_crater\_{CRATER FID}.csv* : list of the azimuthal TVT positions

│   │   ├── images  
│   │   │   ├── *image\_{POS INDEX}\_crater\_{CRATER FID}.npy* : image channels, TVT  

│   │   ├── masks  
│   │   │   ├── *mask\_{POS INDEX}\_crater\_{CRATER FID}.npy* : manual ejecta mask, TVT  

│   │   ├── distances  
│   │   │   ├── *distance\_{POS INDEX}\_crater\_{CRATER FID}.npy* : distance channel, TVT  

│   │   ├── eval_images  
│   │   │   ├── *eval_image\_{EVAL POS INDEX}\_crater\_{CRATER FID}.npy* : image channels, mapping  

│   │   ├── eval_distances  
│   │   │   ├── *eval_distance\_{EVAL POS INDEX}\_crater\_{CRATER FID}.npy* : distance channel, mapping  

│   │   ├── buffers  
│   │   │   ├── *az_crater\_{CRATER FID}.tif* : mosaic portion around crater to extract annulus value for metadata. 

**Mosaics**  
├── data  
│   ├── mosaics  
│   │   ├── *cylindrical_mosaic.tif* : the Enhanced Color mosaic, cylindrical projection  
│   │   ├── *azimuthal_mosaic_crater\_{CRATER FID}.tif* : Enhanced Color mosaic, azimuthal projection centered on crater  

**MAN: the manually-mapped craters**  
├── data  
│   ├── man_data  
│   │   ├── *az_manual\_{CRATER NAME}.tif* : the azimuthal global ejecta mask  
│   │   ├── *manual_FID_map.tif* : a cylindrical map distinguishing the ejecta of different craters  
│   │   ├── *analysis_hm.csv* : analysis of crater patterns  

<br/>

### Results

**EJCONN: the Connection model**  
├── results  
│   ├── conn_results  
│   │   ├── version_32 : final version  
│   │   │   ├── *best-checkpoint_test-set-61_iteration\-{ITERATION NUMBER}.ckpt* : the model weights for a Train-Val split  
│   │   │   ├── *best-norm_test-set-61.npy* : the normalization for the input data  
│   │   │   ├── *test_metrics.csv* : training run data and tests on the TVT tiles (not used for evaluation)  
│   │   │   ├── conn_positions  
│   │   │   │   ├── *conn-pos_test-set-61_crater\-{CRATER FID}.csv* : the connected tiles for a given crater  
│   │   ├── version_{VERSION NUMBER}   

**EJSEG: the Segmentation model**  
├── results  
│   ├── ejc_results  
│   │   ├── version_200 : final version  
│   │   │   ├── *best-checkpoint_test-set-61_iteration\-{ITERATION NUMBER}.ckpt* : the model weights for a Train-Val split  
│   │   │   ├── *best-norm_test-set-61.npy* : the normalization for the input data  
│   │   │   ├── *eval_metrics.csv* : evaluation metrics  
│   │   │   ├── *test_metrics.csv* : training run data and tests on the TVT tiles (not used for evaluation)  
│   │   ├── version_{VERSION NUMBER}   

**Ejecta map**  
├── results  
│   ├── output_masks  
│   │   ├── ejc-version-200_conn-version-32_test-set-61_x20 : final version  
│   │   │   ├── *cylindrical_mask_crater\_{CRATER FID}.tif* : the cylindrical ejecta mask for a given crater  
│   │   │   ├── *ml_FID_map_all.tif* : a cylindrical map distinguishing the ejecta of different craters  
│   │   │   ├── *ejecta-map_all.tif* : the color composite global map  
│   │   │   ├── *analysis_overlaps.csv* : analysis of crater patterns  
│   │   │   ├── *analysis_clean.csv* : analysis of crater patterns with overlaps removed  

<br/>

### Log
The log files are usually temporary, deleted at the end of a training run. But they can be preserved (`train_run_config.cfg`: `RM_LOG=0`) to save the weights (.ckpt files) of each training iteration (not doable for all runs, but for the most important ones).

**EJCONN: the Connection model**  
├── log  
│   ├── conn_log  
│   │   ├── version\_{VERSION NUMBER}  
│   │   │   ├── checkpoints  
│   │   │   │   ├── *best-checkpoint_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.ckpt* : the model weights for a Train-Val split  
│   │   │   ├── norms  
│   │   │   │   ├── *norm_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.ckpt* : the normalization for a Train-Val split  
│   │   │   ├── positions  
│   │   │   │   ├── *eval-pos_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.csv* : connected mapping tiles for a Train-Val split  
│   │   │   ├── tests  
│   │   │   │   ├── *results_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.json* : training run data and tests on the TVT tiles  

**EJSEG: the Segmentation model**  
├── log  
│   ├── ejc_log  
│   │   ├── version\_{VERSION NUMBER}  
│   │   │   ├── checkpoints  
│   │   │   │   ├── *best-checkpoint_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.ckpt* : the model weights for a Train-Val split  
│   │   │   ├── norms  
│   │   │   │   ├── *norm_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.ckpt* : the normalization for a Train-Val split  
│   │   │   ├── tests  
│   │   │   │   ├── *eval_conn-version\-{EJCONN VERSION}\_test-set\-{TV-TEST SPLIT}_iteration\-{ITERATION NUMBER}.json* : evaluation metrics  
│   │   │   │   ├── *results_test-set\-{TV-TEST SPLIT}\_iteration\-{ITERATION NUMBER}.json* : training run data and tests on the TVT tiles  

## Dependencies
The scripts were executed with Python 3.10. The following packages are required.
```
numpy
numexpr
matplotlib
scipy
pandas

geopandas
shapely
pyproj

xarray
rioxarray
rasterio
opencv-python
imgaug
cairosvg

torch
lightning
timm
segmentation-models-pytorch
torchgeo
terratorch
scikit-learn
```

The neural network training must be performed on an NVIDIA GPU (tried on Tesla P40, Tesla V100-PCIE-32GB, GeForce RTX 4070 Laptop). Compatibility of Python packages and CUDA version should be ascertained. 
