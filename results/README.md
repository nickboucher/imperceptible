# Results

The results of our experiments can be found as a Python pickle in `adversarial-examples.pkl` and as a JSON file in `adversarial-examples.json`. The Python pickle and JSON files contain the same data and are both included to provide multiple formats for convenient consumption. The structure of these files is specified by the `../experiments/experiment.py` tool, but from a high level follows the nested dictionary pattern of `experiment_name -> budget -> input_id` (except for targeted attacks, which add one additional innermost nested dictionary level for `... -> target_class`).

Due to GitHub file size limits, the experimental results for targeted NER attacks are stored separately as a Python pickle in `adversarial-examples-targeted-ner.pkl` and as a JSON file in `adversarial-examples-targeted-ner.json`. For the same reason, targeted sentiment analysis (emotion dataset) attacks are stored separately as a Python pickle in `adversarial-examples-targeted-emotion.pkl` and as a JSON file in `adversarial-examples-targeted-emotion.json`

The results of our experiments using OCR as a defense against the generated adversarial examples can be in `ocr-defense.pkl` and `ocr-defense.json`. The structure of these files follows the same pattern as the adversarial-examples files, although the specific file structure details can be found in `../notebooks/OCR Defense.ipynb`.

The notebook `Figures.ipynb` was used to generate all of the figures included in the *Bad Characters* paper using `adversarial-examples.pkl`, with the exception of the OCR defense graph which is generated in `../notebooks/OCR Defense.ipynb`.
