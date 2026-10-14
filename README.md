# Did the tutor notice the mistake?

An audit of our team's submission to the [BEA 2025 Shared Task](https://sig-edu.org/sharedtask/2025)
(Track 1, Mistake Identification), and a stronger system built from what the audit found.

**Report: [aidenn8.github.io/bea2025-mistake-id](https://aidenn8.github.io/bea2025-mistake-id/)**

Given the end of a maths tutoring dialogue in which a student has just made a mistake, and one tutor's
next reply, the task is to say whether the reply recognises the mistake: *Yes*, *To some extent* or
*No*. Systems are ranked by macro-F1 on a hidden test set.

| System | Cross-validated macro-F1 | Official test macro-F1 |
|---|---|---|
| ALA, our 2025 submission (BERT), retrained | 0.576 (0.606 with the new decision rule) | 0.636 (28th of 44) |
| New system (four-backbone ensemble + decision rule) | 0.669 (95% interval 0.641-0.698) | pending (post-competition server) |

Cross-validation is grouped by conversation (see the report for why that matters here). The best single model,
MPNet with the same recipe and rule, scores 0.663; the ensemble's extra 0.007 is within noise.

## Reproducing the results

Python 3.12. The numbers in the report were produced on a 16 GB Apple M2 (PyTorch MPS backend); a CUDA
GPU also works and is much faster.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_data.py        # official files, checked against SHA-256
.venv/bin/python -m pytest -q tests              # data, splits, metrics, encoding, submission format
.venv/bin/python scripts/baselines.py            # majority, tutor identity, TF-IDF baselines
PY=.venv/bin/python bash scripts/queue.sh queue.txt   # every fine-tuning run (about a day on an M2)
.venv/bin/python scripts/analyze.py              # results/summary.json
.venv/bin/python scripts/make_figures.py         # docs/images/
.venv/bin/python scripts/make_submission.py out/ --labels results/test_predictions/final.npy
```

`scripts/run_cv.py CONFIG --split grouped` trains one configuration on each of the five
conversation-grouped folds and saves held-out and test-set probabilities. Interrupted runs resume
from the last finished fold.

## Data

The MRBench data are released by the shared-task organisers under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and are downloaded from
[their repository](https://github.com/kaushal0494/UnifyingAITutorEvaluation) rather than redistributed
here. If you use them, cite:

- Kochmar et al. 2025. *Findings of the BEA 2025 Shared Task on Pedagogical Ability Assessment of
  AI-powered Tutors.* BEA Workshop, ACL 2025.
- Maurya et al. 2025. *Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability
  Assessment of LLM-Powered AI Tutors.* NAACL 2025.

## Credits

The 2025 submission (team ALA) was a team effort. The audit, the evaluation protocol and the new
system in this repository are my own work. Code is MIT-licensed; the fonts in `tools/fonts/` (IBM Plex Mono, Inter) are
under the SIL Open Font License.
