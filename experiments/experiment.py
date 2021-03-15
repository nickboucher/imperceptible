#!/usr/bin/env python3
try:
  import sys
  import torch
  import pickle
  import json
  import numpy as np
  from bs4 import BeautifulSoup
  from timeit import timeit
  from abc import ABC
  from typing import List, Tuple, Callable, Dict
  from fairseq.hub_utils import GeneratorHubInterface
  from scipy.optimize import NonlinearConstraint, differential_evolution
  from textdistance import levenshtein
  from tqdm.auto import tqdm, trange
  from argparse import ArgumentParser
  from sacrebleu import corpus_bleu
  from time import process_time
  from torch.nn.functional import softmax

  # --- Constants ---

  # Zero width space
  ZWSP = chr(0x200B)
  # Zero width joiner
  ZWJ = chr(0x200D)
  # Unicode Bidi override characters
  PDF = chr(0x202C)
  LRE = chr(0x202A)
  RLE = chr(0x202B)
  LRO = chr(0x202D)
  RLO = chr(0x202E)
  PDI = chr(0x2069)
  LRI = chr(0x2066)
  RLI = chr(0x2067)
  # Backspace character
  BKSP = chr(0x8)
  # Delete character
  DEL = chr(0x7F)
  # Carriage return character
  CR = chr(0xD)

  # Retrieve Unicode Intentional homoglyph characters
  intentionals = dict()
  with open("intentional.txt", "r") as f:
    for line in f.readlines():
      if len(line.strip()):
        if line[0] != '#':
          line = line.replace("#*", "#")
          _, line = line.split("#", maxsplit=1)
          if line[3] not in intentionals:
            intentionals[line[3]] = []
          intentionals[line[3]].append(line[7])

  label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  with open('multinli_1.0/multinli_1.0_dev_matched.jsonl', 'r') as f:
    mnli_test = []
    for jline in f.readlines():
      line = json.loads(jline)
      if line['gold_label'] in label_map:
        mnli_test.append(line)

except Exception as ex:
  print("Unable to initiate. Please ensure that you have already run `setup.sh` in this environment.")
  # sys.exit(ex)
  raise ex


# --- Classes ---

class Swap():
    """Represents swapped elements in a string of text."""
    def __init__(self, one, two):
        self.one = one
        self.two = two
    
    def __repr__(self):
        return f"Swap({self.one}, {self.two})"

    def __eq__(self, other):
        return self.one == other.one and self.two == other.two

    def __hash__(self):
        return hash((self.one, self.two))


class Objective(ABC):
  """ Abstract class representing objectives for scipy's genetic algorithms."""

  def __init__(self, model: GeneratorHubInterface, input: str, ref_translation: str, max_perturbs: int, distance: Callable[[str,str],int]):
    if not model:
      raise ValueError("Must supply model.")
    if not input:
      raise ValueError("Must supply input.")

    self.model: GeneratorHubInterface = model
    self.input: str = input
    self.ref_translation = ref_translation
    self.max_perturbs: int = max_perturbs
    self.distance: Callable[[str,str],int] = distance
    self.output = self.model.translate(self.input)

  def objective(self) -> Callable[[List[float]], float]:
    def _objective(perturbations: List[float]) -> float:
      candidate: str = self.candidate(perturbations)
      translation: str = self.model.translate(candidate)
      return -self.distance(self.output, translation)
    return _objective

  def differential_evolution(self, verbose=False, maxiter=60, popsize=32, polish=False) -> str:
    start = process_time()
    result = differential_evolution(self.objective(), self.bounds(),
                                    disp=verbose, maxiter=maxiter,
                                    popsize=popsize, polish=polish)
    end = process_time()
    candidate = self.candidate(result.x)
    return  {
              'adv_example': candidate,
              'adv_example_enc': result.x,
              'input_translation_distance': -result.fun,
              'ref_translation_distance': self.distance(candidate, self.ref_translation),
              'input': self.input,
              'input_translation': self.output,
              'adv_translation': self.model.translate(candidate),
              'ref_translation': self.ref_translation,
              'ref_bleu': corpus_bleu(candidate, self.ref_translation).score,
              'input_bleu': corpus_bleu(candidate, self.input).score,
              'adv_generation_time': end - start,
              'budget': self.max_perturbs,
              'maxiter': maxiter,
              'popsize': popsize
            }

  def bounds(self) -> List[Tuple[float, float]]:
    raise NotImplementedError()

  def candidate(self, perturbations: List[float]) -> str:
    raise NotImplementedError()


class InvisibleCharacterObjective(Objective):
  """Class representing an Objective which injects invisible characters."""

  def __init__(self, model: GeneratorHubInterface, input: str, ref_translation: str, max_perturbs: int = 25, invisible_chrs: List[str] = [ZWJ,ZWSP], distance: Callable[[str,str],int] = levenshtein.distance, **kwargs):
    super().__init__(model, input, ref_translation, max_perturbs, distance)
    self.invisible_chrs: List[str] = invisible_chrs

  def bounds(self) -> List[Tuple[float, float]]:
    return [(0,len(self.invisible_chrs)-1), (-1, len(self.input)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]
    for i in range(0, len(perturbations), 2):
      inp_index = natural(perturbations[i+1])
      if inp_index >= 0:
        inv_char = self.invisible_chrs[natural(perturbations[i])]
        candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
    return ''.join(candidate)


class HomoglyphObjective(Objective):

  def __init__(self, model: GeneratorHubInterface, input: str, ref_translation: str, max_perturbs=None, distance: Callable[[str,str],int] = levenshtein.distance, homoglyphs: Dict[str,List[str]] = intentionals, **kwargs):
    super().__init__(model, input, ref_translation, max_perturbs, distance)
    if not self.max_perturbs:
      self.max_perturbs = len(self.input)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1, len(self.glyph_map)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]  
    for perturb in map(natural, perturbations):
      if perturb >= 0:
        i, char = self.glyph_map[perturb]
        candidate[i] = char
    return ''.join(candidate)


class ReorderObjective(Objective):

  def __init__(self, model: GeneratorHubInterface, input: str, ref_translation: str, max_perturbs: int = 50, distance: Callable[[str,str],int] = levenshtein.distance, **kwargs):
    super().__init__(model, input, ref_translation, max_perturbs, distance)

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1,len(self.input)-1)] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    def swaps(els) -> str:
      res = ""
      for el in els:
          if isinstance(el, Swap):
              res += swaps([LRO, LRI, RLO, LRI, el.one, PDI, LRI, el.two, PDI, PDF, PDI, PDF])
          elif isinstance(el, str):
              res += el
          else:
              for subel in el:
                  res += swaps([subel])
      return res

    _candidate = [char for char in self.input]
    for perturb in map(natural, perturbations):
      if perturb >= 0 and len(_candidate) >= 2:
        perturb = min(perturb, len(_candidate) - 2)
        _candidate = _candidate[:perturb] + [Swap(_candidate[perturb+1], _candidate[perturb])] + _candidate[perturb+2:]

    return swaps(_candidate)


class DeletionObjective(Objective):
  """Class representing an Objective which injects deletion control characters."""

  def __init__(self, model: GeneratorHubInterface, input: str, max_perturbs: int = 100, distance: Callable[[str,str],int] = levenshtein.distance, del_chr: str = BKSP, ins_chr_min: str = '!', ins_chr_max: str = '~', **kwargs):
    super().__init__(model, input, max_perturbs, distance)
    self.del_chr: str = del_chr
    self.ins_chr_min: str = ins_chr_min
    self.ins_chr_max: str = ins_chr_max

  def bounds(self) -> List[Tuple[float, float]]:
    return [(-1,len(self.input)-1), (ord(self.ins_chr_min),ord(self.ins_chr_max))] * self.max_perturbs

  def candidate(self, perturbations: List[float]) -> str:
    candidate = [char for char in self.input]
    for i in range(0, len(perturbations), 2):
      idx = natural(perturbations[i])
      char = chr(natural(perturbations[i+1]))
      candidate = candidate[:idx] + [char, self.del_chr] + candidate[idx:]
      for j in range(i,len(perturbations), 2):
        perturbations[j] += 2
    return ''.join(candidate)


class MnliObjective():

  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, max_perturbs: int):
    if not model:
      raise ValueError("Must supply model.")
    if not input:
      raise ValueError("Must supply input.")
    if not hypothesis:
      raise ValueError("Must supply hypothesis.")
    if label == None:
      raise ValueError("Must supply label.")
    self.model: GeneratorHubInterface = model
    self.input: str = input
    self.hypothesis: str = hypothesis
    self.label: int = label
    self.max_perturbs: int = max_perturbs

  def objective(self) -> Callable[[List[float]], float]:
    def _objective(perturbations: List[float]) -> float:
      candidate: str = self.candidate(perturbations)
      tokens = self.model.encode(candidate, self.hypothesis)
      predict = self.model.predict('mnli', tokens)
      if predict.argmax() != self.label:
        return -np.inf
      else:
        return predict.cpu().detach().numpy()[0][self.label]
    return _objective

  def differential_evolution(self, verbose=False, maxiter=3, popsize=32, polish=False) -> str:
    start = process_time()
    result = differential_evolution(self.objective(), self.bounds(),
                                    disp=verbose, maxiter=maxiter,
                                    popsize=popsize, polish=polish)
    end = process_time()
    candidate = self.candidate(result.x)
    tokens = self.model.encode(candidate, self.hypothesis)
    predict = self.model.predict('mnli', tokens)
    inp_tokens = self.model.encode(self.input, self.hypothesis)
    inp_predict = self.model.predict('mnli', tokens)
    return  {
              'adv_example': candidate,
              'adv_example_enc': result.x,
              'input': self.input,
              'hypothesis': self.hypothesis,
              'correct_label_index': self.label,
              'adv_predictions': predict.cpu().detach().numpy()[0],
              'input_prediction': inp_predict.cpu().detach().numpy()[0],
              'adv_prediction_correct': predict.argmax().item() == self.label,
              'input_prediction_correct': inp_predict.argmax().item() == self.label,
              'adv_generation_time': end - start,
              'budget': self.max_perturbs,
              'maxiter': maxiter,
              'popsize': popsize
            }


class InvisibleCharacterMnliObjective(MnliObjective, InvisibleCharacterObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, max_perturbs: int = 10, invisible_chrs: List[str] = [ZWJ,ZWSP], **kwargs):
    super().__init__(model, input, hypothesis, label, max_perturbs)
    self.invisible_chrs = invisible_chrs


class HomoglyphMnliObjective(MnliObjective, HomoglyphObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, max_perturbs: int = 10, homoglyphs: Dict[str,List[str]] = intentionals, **kwargs):
    super().__init__(model, input, hypothesis, label, max_perturbs)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)


class ReorderMnliObjective(MnliObjective, ReorderObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, max_perturbs: int = 10, **kwargs):
    super().__init__(model, input, hypothesis, label, max_perturbs)


class DeletionMnliObjective(MnliObjective, DeletionObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, max_perturbs: int = 10, del_chr: str = BKSP, ins_chr_min: str = '!', ins_chr_max: str = '~', **kwargs):
    super().__init__(model, input, hypothesis, label, max_perturbs)
    self.del_chr: str = del_chr
    self.ins_chr_min: str = ins_chr_min
    self.ins_chr_max: str = ins_chr_max


class HomoglyphSpongeObjective(HomoglyphObjective):

  def objective(self) -> Callable[[List[float]], float]:
    def _objective(perturbations: List[float]) -> float:
      candidate: str = self.candidate(perturbations)
      return -1 * timeit(lambda: self.model.translate(candidate), number=1)
    return _objective


class MnliTargetedObjective(MnliObjective):

  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label: int, target: int, max_perturbs: int):
    super().__init__(model, input, hypothesis, label, max_perturbs)
    self.target = target

  def objective(self) -> Callable[[List[float]], float]:
      def _objective(perturbations: List[float]) -> float:
        candidate: str = self.candidate(perturbations)
        tokens = self.model.encode(candidate, self.hypothesis)
        predict = self.model.predict('mnli', tokens)
        return -softmax(predict, dim=1).cpu().detach().numpy()[0][self.target]
      return _objective

  def differential_evolution(self, verbose=False, maxiter=3, popsize=32, polish=False) -> str:
    start = process_time()
    result = differential_evolution(self.objective(), self.bounds(),
                                    disp=verbose, maxiter=maxiter,
                                    popsize=popsize, polish=polish)
    end = process_time()
    candidate = self.candidate(result.x)
    tokens = self.model.encode(candidate, self.hypothesis)
    predict = self.model.predict('mnli', tokens)
    probs = softmax(predict, dim=1).cpu().detach().numpy()[0]
    selection = probs.argmax().item()
    inp_tokens = self.model.encode(self.input, self.hypothesis)
    inp_predict = self.model.predict('mnli', tokens)
    inp_probs = softmax(inp_predict, dim=1).cpu().detach().numpy()[0]
    inp_selection = inp_probs.argmax().item()
    return {
      'adv_example': candidate,
      'adv_example_enc': result.x,
      'input': self.input,
      'hypothesis': self.hypothesis,
      'golden_label': self.label,
      'adv_predictions': probs,
      'input_prediction': inp_probs,
      'adv_target_success': selection == self.target,
      'adv_golden_correct': selection == self.label,
      'input_golden_correct': inp_selection == self.label,
      'target_label': self.target,
      'adv_selected_label': selection,
      'input_selected_label': inp_selection,
      'adv_generation_time': end - start,
      'budget': self.max_perturbs,
      'maxiter': maxiter,
      'popsize': popsize
    }

class InvisibleCharacterTargetedMnliObjective(MnliTargetedObjective, InvisibleCharacterObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, target: int, max_perturbs: int = 10, invisible_chrs: List[str] = [ZWJ,ZWSP], **kwargs):
    super().__init__(model, input, hypothesis, label, target, max_perturbs)
    self.invisible_chrs = invisible_chrs


class HomoglyphTargetedMnliObjective(MnliTargetedObjective, HomoglyphObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, target: int, max_perturbs: int = 10, homoglyphs: Dict[str,List[str]] = intentionals, **kwargs):
    super().__init__(model, input, hypothesis, label, target, max_perturbs)
    self.homoglyphs = homoglyphs
    self.glyph_map = []
    for i, char in enumerate(self.input):
      if char in self.homoglyphs:
        charmap = self.homoglyphs[char]
        charmap = list(zip([i] * len(charmap), charmap))
        self.glyph_map.extend(charmap)


class ReorderTargetedMnliObjective(MnliTargetedObjective, ReorderObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, target: int, max_perturbs: int = 10, **kwargs):
    super().__init__(model, input, hypothesis, label, target, max_perturbs)


class DeletionTargetedMnliObjective(MnliTargetedObjective, DeletionObjective):
  
  def __init__(self, model: GeneratorHubInterface, input: str, hypothesis: str, label:int, target: int, max_perturbs: int = 10, del_chr: str = BKSP, ins_chr_min: str = '!', ins_chr_max: str = '~', **kwargs):
    super().__init__(model, input, hypothesis, label, target, max_perturbs)
    self.del_chr = del_chr
    self.ins_chr_min: str = ins_chr_min
    self.ins_chr_max: str = ins_chr_max


class MnliTargetedNoLogitsObjective(MnliTargetedObjective):

  def objective(self) -> Callable[[List[float]], float]:
      def _objective(perturbations: List[float]) -> float:
        candidate: str = self.candidate(perturbations)
        tokens = self.model.encode(candidate, self.hypothesis)
        predict = self.model.predict('mnli', tokens)
        if predict.argmax().item() == self.target:
          return -np.inf
        else:
          return np.inf
      return _objective


class InvisibleCharacterTargetedMnliNoLogitsObjective(MnliTargetedNoLogitsObjective, InvisibleCharacterTargetedMnliObjective):
  pass


class HomoglyphTargetedMnliNoLogitsObjective(MnliTargetedNoLogitsObjective, HomoglyphTargetedMnliObjective):
  pass


class ReorderTargetedMnliNoLogitsObjective(MnliTargetedNoLogitsObjective, ReorderTargetedMnliObjective):
  pass


class DeletionTargetedMnliNoLogitsObjective(MnliTargetedNoLogitsObjective, DeletionTargetedMnliObjective):
  pass


# --- Helper Functions ---

def some(*els):
    """Returns the arguments as a tuple with Nones removed."""
    return tuple(filter(None, tuple(els)))

def swaps(chars: str) -> set:
    """Generates all possible swaps for a string."""
    def pairs(chars, pre=(), suf=()):
        orders = set()
        for i in range(len(chars)-1):
            prefix = pre + tuple(chars[:i])
            suffix = suf + tuple(chars[i+2:])
            swap = Swap(chars[i+1], chars[i])
            pair = some(prefix, swap, suffix)
            orders.add(pair)
            orders.update(pairs(suffix, pre=some(prefix, swap)))
            orders.update(pairs(some(prefix, swap), suf=suffix))
        return orders
    return pairs(chars) | {tuple(chars)}

def unswap(el: tuple) -> str:
    """Reverts a tuple of swaps to the original string."""
    if isinstance(el, str):
        return el
    elif isinstance(el, Swap):
        return unswap((el.two, el.one))
    else:
        res = ""
        for e in el:
            res += unswap(e)
        return res

def uniswap(els):
    res = ""
    for el in els:
        if isinstance(el, Swap):
            res += uniswap([LRO, LRI, RLO, LRI, el.one, PDI, LRI, el.two, PDI, PDF, PDI, PDF])
        elif isinstance(el, str):
            res += el
        else:
            for subel in el:
                res += uniswap([subel])
    return res

def natural(x: float) -> int:
    """Rounds float to the nearest natural number (positive int)"""
    return max(0, round(float(x)))

def load_source(num_examples):
  # Build source and target mappings for BLEU scoring
  source = dict()
  target = dict()
  with open('newstest2014-fren-src.en.sgm', 'r') as f:
    source_doc = BeautifulSoup(f, 'html.parser')
  with open('newstest2014-fren-ref.fr.sgm', 'r') as f:
    target_doc = BeautifulSoup(f, 'html.parser')
  i = 0
  for doc in source_doc.find_all('doc'):
    if i < num_examples:
      source[str(doc['docid'])] = dict()
      for seg in doc.find_all('seg'):
        if i < num_examples:
          source[str(doc['docid'])][str(seg['id'])] = str(seg.string)
          i += 1
  for docid, doc in source.items():
    target[docid] = dict()
    for segid in doc:
      node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
      target[docid][segid] = str(node.string)
  return source, target, i

def experiment(model, objective, source, target, min_perturb, max_perturb, file, maxiter, popsize, n_examples, label):
  perturbs = { label: { '0': dict() } }
  for docid, doc in source.items():
    perturbs[label]['0'][docid] = {}
    for segid, seg in doc.items():
      ref = target[docid][segid]
      output = model.translate(seg)
      perturbs[label]['0'][docid][segid] = {
            'adv_example': seg,
            'adv_example_enc': [],
            'input_translation_distance': levenshtein.distance(seg, seg),
            'ref_translation_distance': levenshtein.distance(seg, ref),
            'input': seg,
            'input_translation': output,
            'adv_translation': output,
            'ref_translation': ref,
            'ref_bleu': corpus_bleu(seg, ref).score,
            'input_bleu': corpus_bleu(seg, seg).score,
            'adv_generation_time': 0,
            'budget': 0,
            'maxiter': maxiter,
            'popsize': popsize
          }
  with tqdm(total=n_examples*(max_perturb-min_perturb+1), desc="Adv. Examples") as pbar:
    for i in range(min_perturb, max_perturb+1):
      perturbs[label][str(i)] = dict()
      for docid, doc in source.items():
        perturbs[label][str(i)][docid] = dict()
        for segid, seg in doc.items():
          ref = target[docid][segid]
          perturbs[label][str(i)][docid][segid] = objective(en2fr, seg, ref, max_perturbs=i).differential_evolution(maxiter=maxiter, popsize=popsize)
          with open(file, 'wb') as f:
            pickle.dump(perturbs, f)
          pbar.update(1)

def mnli_experiment(model, objective, data, file, min_budget, max_budget, maxiter, popsize, exp_label):
  perturbs = { exp_label: { '0': dict() } }
  for test in data:
    tokens = model.encode(test['sentence1'], test['sentence2'])
    predict = model.predict('mnli', tokens)
    predictions = predict.cpu().detach().numpy()[0]
    label = label_map[test['gold_label']]
    correct = predict.argmax().item() == label
    perturbs[exp_label]['0'][test['pairID']] = {
        'adv_example': test['sentence1'],
        'adv_example_enc': [],
        'input': test['sentence1'],
        'hypothesis': test['sentence2'],
        'correct_label_index': label,
        'adv_predictions': predictions,
        'input_prediction': predictions,
        'adv_prediction_correct': correct,
        'input_prediction_correct': correct,
        'adv_generation_time': 0,
        'budget': 0,
        'maxiter': maxiter,
        'popsize': popsize
      }
  with tqdm(total=len(data)*(max_budget-min_budget+1), desc="Adv. Examples") as pbar:
    for budget in range(min_budget, max_budget+1):
      perturbs[exp_label][str(budget)] = dict()
      for test in data:
        obj = objective(mnli, test['sentence1'], test['sentence2'], label_map[test['gold_label']], budget)
        example = obj.differential_evolution(maxiter=maxiter, popsize=popsize)
        perturbs[exp_label][str(budget)][test['pairID']] = example
        with open(file, 'wb') as f:
          pickle.dump(perturbs, f)
        pbar.update(1)


def mnli_targeted_experiment(objective, model, inputs, file, min_budget, max_budget, maxiter, popsize, exp_label):
  perturbs = { exp_label: { '0': dict() } }
  for test in data:
    tokens = model.encode(test['sentence1'], test['sentence2'])
    predict = model.predict('mnli', tokens)
    probs = softmax(predict, dim=1).cpu().detach().numpy()[0]
    selection = probs.argmax().item()
    label = label_map[test['gold_label']]
    correct = predict.argmax().item() == label
    perturbs[exp_label]['0'][test['pairID']] = dict()
    for target in range(len(label_map)):
      perturbs[exp_label]['0'][test['pairID']][str(target)] = {
            'adv_example': test['sentence1'],
            'adv_example_enc': [],
            'input': test['sentence1'],
            'hypothesis': test['sentence2'],
            'golden_label': label,
            'adv_predictions': probs,
            'input_prediction': probs,
            'adv_target_success': selection == target,
            'adv_golden_correct': selection == label,
            'input_golden_correct': selection == label,
            'target_label': target,
            'adv_selected_label': selection,
            'input_selected_label': selection,
            'adv_generation_time': 0,
            'budget': 0,
            'maxiter': maxiter,
            'popsize': popsize
          }
  with tqdm(total=len(inputs)*(max_budget-min_budget+1)*len(label_map), desc="Adv. Examples") as pbar:
    for budget in range(min_budget, max_budget+1):
      perturbs[exp_label][str(budget)] = dict()
      for input in inputs:
        perturbs[exp_label][str(budget)][input['pairID']] = dict()
        for target in range(len(label_map)):
          obj = objective(model, input['sentence1'], input['sentence2'], label_map[input['gold_label']], target, budget)
          example = obj.differential_evolution(verbose=False, maxiter=maxiter, popsize=popsize)
          perturbs[exp_label][str(budget)][input['pairID']][str(target)] = example
          with open(file, 'wb') as f:
            pickle.dump(perturbs, f)
          pbar.update(1)

def load_en2fr(cpu):
  # Load pre-trained translation model
  print("Loading EN->FR translation model.")
  en2fr = torch.hub.load('pytorch/fairseq',
                        'transformer.wmt14.en-fr',
                        tokenizer='moses',
                        bpe='subword_nmt',
                        verbose=False).eval()
  if cpu:
    en2fr.cpu()
  else:
    en2fr.cuda()
  print("Model loaded successfully.")
  return en2fr

def load_mnli(cpu):
  # Load pre-trained MNLI model
  print("Loading MNLI classification model.")
  mnli = torch.hub.load('pytorch/fairseq',
                        'roberta.large.mnli',
                        verbose=False).eval()
  if cpu:
    mnli.cpu()
  else:
    mnli.cuda()
  print("Model loaded successfully.")
  return mnli


# -- CLI ---

if __name__ == '__main__':

  parser = ArgumentParser(description='Adversarial NLP Experiments.')
  technique = parser.add_mutually_exclusive_group(required=True)
  technique.add_argument('-i', '--invisible-chars', action='store_true', help="Use invisible character perturbations.")
  technique.add_argument('-g', '--homoglyphs', action='store_true', help="Use homoglyph perturbations.")
  technique.add_argument('-r', '--reorderings', action='store_true', help="Use reordering perturbations.")
  technique.add_argument('-d', '--deletions', action='store_true', help="Use deletion perturbations.")
  task = parser.add_mutually_exclusive_group(required=True)
  task.add_argument('-t', '--translation', action='store_true', help="Target translation task (EN->FR).")
  task.add_argument('-m', '--mnli', action='store_true', help="Target MNLI task (Roberta).")
  parser.add_argument('-c', '--cpu', action='store_true', default=True, help="Use CPU for ML inference instead of CUDA.")
  parser.add_argument('pkl_file', help="File to contain Python pickled output.")
  parser.add_argument('-n', '--num-examples', type=int, default=500, help="Number of adversarial examples to generate.")
  parser.add_argument('-l', '--min-perturbs', type=int, default=1, help="The lower bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-u', '--max-perturbs', type=int, default=5, help="The upper bound (inclusive) of the perturbation budget range.")
  parser.add_argument('-a', '--maxiter', type=int, default=10, help="The maximum number of iterations in the genetic algorithm.")
  parser.add_argument('-p', '--popsize', type=int, default=32, help="The size of the population in he genetic algorithm.")
  targeted = parser.add_mutually_exclusive_group()
  targeted.add_argument('-x', '--targeted', action='store_true', help="Perform a targeted attack.")
  targeted.add_argument('-X', '--targeted-no-logits', action='store_true', help="Perform a targeted attack without access to inference result logits.")
  args = parser.parse_args()

  if args.translation:
    if args.targeted:
      print("Targeted attacks for translation do not exist.")
      sys.exit(1)

    en2fr = load_en2fr(args.cpu)
    source, target, n_examples = load_source(args.num_examples)
    print(f"Loaded {n_examples} strings from corpus.")

    if args.invisible_chars:
      print("Starting invisible characters translation experiment.")
      objective = InvisibleCharacterObjective
      label = "translation_invisibles"
    elif args.homoglyphs:
      print("Starting homoglyphs translation experiment.")
      objective = InvisibleCharacterObjective
      label = "translation_homoglyphs"
    elif args.reorderings:
      print("Starting reorderings translation experiment.")
      objective = InvisibleCharacterObjective
      label = "translation_reorderings"
    elif args.deletions:
      print("Starting deletions translation experiment.")
      objective = InvisibleCharacterObjective
      label = "translation_deletions"

    experiment(en2fr, objective, source, target, args.min_perturbs, args.max_perturbs, args.pkl_file, args.maxiter, args.popsize, n_examples, label)

  elif args.mnli:
    mnli = load_mnli(args.cpu)
    data = mnli_test[:args.num_examples]
    print(f"Loaded {len(data)} strings from corpus.")

    if args.targeted:
      if args.invisible_chars:
        print(f"Starting invisible characters targeted MNLI experiment.")
        objective = InvisibleCharacterTargetedMnliObjective
        label = "mnli_invisibles_targeted"
      elif args.homoglyphs:
        print(f"Starting homoglyphs targeted MNLI experiment.")
        objective = InvisibleCharacterTargetedMnliObjective
        label = "mnli_homoglyphs_targeted"
      elif args.reorderings:
        print(f"Starting reorderings targeted MNLI experiment.")
        objective = InvisibleCharacterTargetedMnliObjective
        label = "mnli_reorderings_targeted"
      elif args.deletions:
        print(f"Starting deletions targeted MNLI experiment.")
        objective = InvisibleCharacterTargetedMnliObjective
        label = "mnli_deletions_targeted"
      
      mnli_targeted_experiment(objective, mnli, data, args.pkl_file, args.min_perturbs, args.max_perturbs, args.maxiter, args.popsize, label)
    
    elif args.targeted_no_logits:
      if args.invisible_chars:
        print(f"Starting invisible characters targeted MNLI (no logits) experiment.")
        objective = InvisibleCharacterTargetedMnliNoLogitsObjective
        label = "mnli_invisibles_targeted_nologits"
      elif args.homoglyphs:
        print(f"Starting homoglyphs targeted MNLI (no logits) experiment.")
        objective = InvisibleCharacterTargetedMnliNoLogitsObjective
        label = "mnli_homoglyphs_targeted_nologits"
      elif args.reorderings:
        print(f"Starting reorderings targeted MNLI (no logits) experiment.")
        objective = InvisibleCharacterTargetedMnliNoLogitsObjective
        label = "mnli_reorderings_targeted_nologits"
      elif args.deletions:
        print(f"Starting deletions targeted MNLI (no logits) experiment.")
        objective = InvisibleCharacterTargetedMnliNoLogitsObjective
        label = "mnli_deletions_targeted_nologits"
      
      mnli_targeted_experiment(objective, mnli, data, args.pkl_file, args.min_perturbs, args.max_perturbs, args.maxiter, args.popsize, label)
    
    else:
      if args.invisible_chars:
        print(f"Starting invisible characters MNLI experiment.")
        objective = InvisibleCharacterMnliObjective
        label = "mnli_invisibles_untargeted"
      elif args.homoglyphs:
        print(f"Starting homoglyphs MNLI experiment.")
        objective = InvisibleCharacterMnliObjective
        label = "mnli_homoglyphs_untargeted"
      elif args.reorderings:
        print(f"Starting reorderings MNLI experiment.")
        objective = InvisibleCharacterMnliObjective
        label = "mnli_reorderings_untargeted"
      elif args.deletions:
        print(f"Starting deletions MNLI experiment.")
        objective = InvisibleCharacterMnliObjective
        label = "mnli_deletions_untargeted"

      mnli_experiment(mnli, objective, data, args.pkl_file, args.min_perturbs, args.max_perturbs, args.maxiter, args.popsize, label)

print(f"Experiment complete. Results written to {args.pkl_file}.")
