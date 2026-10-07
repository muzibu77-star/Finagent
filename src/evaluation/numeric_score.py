"""Scalar-only TAT numeric normalization and strict JSON result parsing.

Protocol reference: TAT-QA revision 870accc41953dcde885aabeb963d94aabdc0fbc3,
get_answer_str/to_number. This implementation covers the frozen numeric subset,
not multi-span text F1 or the full official evaluator.
"""

import json
import math
import re

SCALES = {'': 1, 'thousand': 1000, 'million': 1000000,
          'billion': 1000000000, 'percent': 0.01}


def disclosed_number(text: str) -> float:
    text = text.strip()
    if not re.fullmatch(r'[$€£]?\(?[+-]?\d[\d,]*(?:\.\d+)?%?\)?', text):
        raise ValueError('expected a single disclosed numeric scalar')
    negative = '(' in text
    percent = '%' in text
    value = float(re.sub(r'[$€£(),%]', '', text))
    value *= -1 if negative else 1
    value = round(value * (0.01 if percent else 1), 4)
    if not math.isfinite(value):
        raise ValueError('nonfinite result')
    return round(value, 4)


def scalar(text: str, scale: str) -> float:
    if scale not in SCALES:
        raise ValueError('unknown numeric scale')
    value = disclosed_number(text)
    return round(value if '%' in text else round(value, 2) * SCALES[scale], 4)


def parse_numeric(text: str) -> dict:
    prediction = json.loads(text.strip())
    if not isinstance(prediction, dict) or set(prediction) != {'answer', 'scale'}:
        raise ValueError('answer and scale fields required')
    answer = prediction['answer']
    if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
        raise ValueError('answer must be a scalar')
    scalar(str(answer), prediction['scale'])
    return prediction


def score(prediction: dict, gold: dict) -> dict:
    answer = gold['answer'][0] if isinstance(gold['answer'], list) else gold['answer']
    expected = scalar(str(answer), gold['scale'])
    text = str(prediction['answer'])
    candidates = {scalar(text, prediction['scale'])}
    if not prediction['scale'] and '%' not in text:
        candidates.add(disclosed_number(text))
    return {'numeric_em': expected in candidates,
            'scale_correct': prediction['scale'] == gold['scale']}
