# Deep learning map of fresh crater ejecta on Mercury
A repository for EJMAP, a deep learning algorithm to map crater ejecta on the surface of Mercury and link them to their crater of origin.

<img src="https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/final_map.png" width="950">

EJMAP consists in two neural network models: EJCONN, a classification model that selects the tiles of a Mercury mosaic where the ejecta from a given crater are located, and EJSEG, a segmentation model which maps the ejecta on those tiles ([explanatory diagram](https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/mapping_procedure.png)).

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

**Planet characteristics**  
├── code  
│   ├── *planet_constants.py* : holds the parameters of the planet. Change to apply to a different planet.

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
│   │   ├── *az_positions_all.csv* : list of all TVT azimuthal tiles, the `test\_{SPLIT CODE}` columns assign them to TV (0) or Test (1).  

**EJCONN: the Connection model**  
├── data  
├── conn_data  
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

├── images
├── log
│   ├── conn_log
│   └── ejc_log
└── results
    ├── conn_results
    │   ├── version_...
    │   │   └── conn_positions
    ├── ejc_results
    │   ├── version_...
    └── output_masks
        └── ejc-version-200_conn-version-32_test-set-61_x20
