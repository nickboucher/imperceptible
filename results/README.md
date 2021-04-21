# Results

The results of our experiments can be found as a Python pickle in `results.pkl` and as a JSON file in `results.json`. The structure of these files is specified by the `../experiments.py` tool, but from a high level follows the nexted dictionary pattern of `experiment_name -> budget -> input_id`.

The notebook `Figures.ipynb` was used to generate all of the figures included in the imperceptible perturbations paper using the pickle file in this directory.
