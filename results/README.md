# Results

The results of our experiments can be found as a Python pickle in `adversarial-examples.pkl` and as a JSON file in `adversarial-examples.json`. The structure of these files is specified by the `../experiments/experiment.py` tool, but from a high level follows the nested dictionary pattern of `experiment_name -> budget -> input_id`.

The results of our experiments using OCR as a defense against the generated adversarial examples can be in `ocr-defense.pkl` and `ocr-defense.json`. The structure of these files follows the same pattern as the adversarial-examples files, although the specific file structure details can be found in `../notebooks/OCR Defense.ipynb`.

The notebook `Figures.ipynb` was used to generate all of the figures included in the *Bad Characters* paper using `adversarial-examples.pkl`, with the exception of the OCR defense graph which is generated in `../notebooks/OCR Defense.ipynb`.
