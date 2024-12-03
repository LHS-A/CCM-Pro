# CCM-Pro
## Progressive Distillation Incremental Learning for Corneal Confocal Microscopy Segmentation

The CORN-Pro dataset can be download from [Zenodo CORN-Pro](https://doi.org/10.5281/zenodo.14263883)

### Running Training

1. Adjust the data path to match the required structure:  
   - `train -> image, label`  
   - `val -> image, label`  
   - `test -> image, label`

2. Begin by training the CNs task. Without specifying the `model_initialization` function, select models `model_S` and `model_T`, then run `main.py` to obtain the optimal segmentation model for CNs.

3. Next, load the optimal weights from the CNs task using the `model_initialization` function. Then, invoke `model_S_adapter` and `model_T_adapter`, and run `main.py` to derive the optimal segmentation model for LCs.

4. Finally, use the same training setup for the LCs task to obtain the optimal segmentation model for the SCs task.

### Testing

Load the optimal weights for the task, and then execute `test.py`.
