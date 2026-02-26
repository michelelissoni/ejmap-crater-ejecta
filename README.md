# ejmap-crater-ejecta
A repository for EJMAP, a deep learning algorithm to map crater ejecta on the surface of Mercury and link them to their crater of origin.

<img src="https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/final_map.png" width="950">

EJMAP consists in two neural network models: EJCONN, a classification model that selects the tiles of a Mercury mosaic where the ejecta from a given crater are located, and EJSEG, a segmentation model which maps the ejecta on those tiles ([explanatory diagram](https://github.com/michelelissoni/ejmap-crater-ejecta/blob/main/images/mapping_procedure.png)).
