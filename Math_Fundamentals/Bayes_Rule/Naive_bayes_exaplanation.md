# Naive Bayes — Full Walkthrough

A complete, worked example of a Multinomial Naive Bayes classifier for text — from training to prediction — explained step by step.

---

## Table of Contents

1. [Overview](#overview)
2. [The Class Structure](#the-class-structure)
3. [Key Concepts](#key-concepts)
   - [The `smoothing` Parameter](#the-smoothing-parameter)
   - [Understanding `defaultdict`](#understanding-defaultdict)
4. [Training Walkthrough](#training-walkthrough)
5. [Prediction Walkthrough](#prediction-walkthrough)
6. [Why Logs?](#why-logs)
7. [Summary](#summary)

---

## Overview

Naive Bayes is a simple, fast classifier for text. It:

- **Learns** by counting how often words appear in each class.
- **Predicts** by scoring a new document against each class and picking the highest.

No gradients. No optimization. Just **counting** and **comparing**.

---

## The Class Structure

```python
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()
```

### What Each Attribute Does

| Attribute | Purpose |
|-----------|---------|
| `smoothing` | Small number added to every word count so no probability is ever zero. |
| `class_counts` | How many documents of each class were seen. Gives `P(class)`. |
| `word_counts` | Nested dict: how often each word appeared in each class. Gives `P(word \| class)`. |
| `class_word_totals` | Total words per class. Denominator for word probabilities. |
| `vocab` | Set of every unique word seen. Needed for smoothing denominator. |

---

## Key Concepts

### The `smoothing` Parameter

**Smoothing** (Laplace smoothing) adds a small number to every word count so **no word ever gets probability zero**.

#### The Problem It Solves

Imagine a word `"lottery"` never appeared in the "ham" class:

```
P("lottery" | ham) = 0 / total = 0
```

Now a new email arrives: *"Meeting about the lottery project"*

```
P(ham) × P("meeting"|ham) × P("lottery"|ham) × P("project"|ham)
= something × something × 0 × something
= 0
```

**One unseen word wipes out the entire prediction.**

#### The Fix

Add `smoothing` to every count:

```
P("lottery" | ham) = (0 + 1) / (total + 1 × vocab_size)
                  = tiny but nonzero
```

#### Choosing a Value

| Value | Effect |
|-------|--------|
| `0` | No smoothing — zeros possible (dangerous) |
| `1.0` | Classic Laplace smoothing (default) |
| `0.1` | Lighter smoothing — trusts data more |
| `10` | Heavy smoothing — pushes toward uniform |

**Rule of thumb:** Use `1.0` unless you have a reason not to.

---

### Understanding `defaultdict`

A normal dict crashes on missing keys:

```python
d = {}
d["spam"] += 1   # ❌ KeyError
```

`defaultdict(factory)` auto-creates missing keys using `factory()`:

```python
from collections import defaultdict

d = defaultdict(int)   # missing keys default to 0
d["spam"] += 1         # works! "spam" created as 0, then becomes 1
```

#### Common Factories

| Factory | Default Value |
|---------|---------------|
| `int` | `0` |
| `float` | `0.0` |
| `str` | `""` |
| `list` | `[]` |
| `set` | `set()` |
| `lambda: defaultdict(int)` | a fresh nested defaultdict |

> **Note:** Pass the function (`int`), not the result (`int()`).

#### Nested `defaultdict`

```python
self.word_counts = defaultdict(lambda: defaultdict(int))
```

This lets you write:

```python
word_counts["spam"]["free"] += 1
```

without any setup. The `lambda` ensures each outer key gets its **own** inner dict (using a shared instance would be a bug).

#### The Gotcha

Accessing a missing key **creates it**:

```python
d = defaultdict(int)
if d["nonexistent"] == 0:
    pass
print(d)   # {'nonexistent': 0}  ← created just by checking!
```

To check safely, use `"key" in d` or `d.get("key", 0)`.

---

## Training Walkthrough

### Training Data

```python
documents = ["Free money now", "Meeting at noon", "Win free cash"]
labels    = ["spam",            "ham",             "spam"]
```

### The Training Loop

```python
def train(self, documents, labels):
    for doc, label in zip(documents, labels):
        self.class_counts[label] += 1
        words = doc.lower().split()
        for word in words:
            self.word_counts[label][word] += 1
            self.class_word_totals[label] += 1
            self.vocab.add(word)
```

### Iteration 1: `("Free money now", "spam")`

- `class_counts["spam"]` → `1`
- Words: `["free", "money", "now"]`
- After processing:

```python
word_counts["spam"]         = {"free": 1, "money": 1, "now": 1}
class_word_totals["spam"]   = 3
vocab                       = {"free", "money", "now"}
```

### Iteration 2: `("Meeting at noon", "ham")`

- `class_counts["ham"]` → `1`
- Words: `["meeting", "at", "noon"]`
- After processing:

```python
word_counts["ham"]          = {"meeting": 1, "at": 1, "noon": 1}
class_word_totals["ham"]    = 3
vocab                       = {"free", "money", "now", "meeting", "at", "noon"}
```

### Iteration 3: `("Win free cash", "spam")`

- `class_counts["spam"]` → `2`
- Words: `["win", "free", "cash"]`
- `"free"` seen again → its count goes from `1` → `2`
- After processing:

```python
word_counts["spam"]         = {"free": 2, "money": 1, "now": 1, "win": 1, "cash": 1}
class_word_totals["spam"]   = 6
vocab                       = {"free", "money", "now", "meeting", "at", "noon", "win", "cash"}
```

### Final Trained State

```python
class_counts      = {"spam": 2, "ham": 1}
word_counts       = {
                      "spam": {"free": 2, "money": 1, "now": 1, "win": 1, "cash": 1},
                      "ham":  {"meeting": 1, "at": 1, "noon": 1}
                    }
class_word_totals = {"spam": 6, "ham": 3}
vocab             = {"free", "money", "now", "meeting", "at", "noon", "win", "cash"}
```

---

## Prediction Walkthrough

### The Prediction Code

```python
def predict(self, document):
    words = document.lower().split()
    total_docs = sum(self.class_counts.values())
    vocab_size = len(self.vocab)
    best_class = None
    best_score = float("-inf")
    for cls in self.class_counts:
        score = math.log(self.class_counts[cls] / total_docs)
        for word in words:
            count = self.word_counts[cls].get(word, 0)
            total = self.class_word_totals[cls]
            score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
        if score > best_score:
            best_score = score
            best_class = cls
    return best_class
```

### Input

Let's classify: **`"free meeting"`**

### Setup

```python
words       = ["free", "meeting"]
total_docs  = 2 + 1 = 3
vocab_size  = 8
best_class  = None
best_score  = float("-inf")
```

### Class 1: `"spam"`

**Prior:**

```
score = log(2 / 3) ≈ -0.405
```

**Word `"free"`:**

```
count = word_counts["spam"]["free"] = 2
total = class_word_totals["spam"] = 6

log((2 + 1) / (6 + 1×8)) = log(3 / 14) ≈ -1.540

score = -0.405 + (-1.540) = -1.945
```

**Word `"meeting"`:**

```
count = word_counts["spam"].get("meeting", 0) = 0   # never seen in spam
total = 6

log((0 + 1) / (6 + 8)) = log(1 / 14) ≈ -2.639

score = -1.945 + (-2.639) = -4.584
```

**Final spam score: `-4.584`**

### Class 2: `"ham"`

**Prior:**

```
score = log(1 / 3) ≈ -1.099
```

**Word `"free"`:**

```
count = word_counts["ham"].get("free", 0) = 0
total = class_word_totals["ham"] = 3

log((0 + 1) / (3 + 8)) = log(1 / 11) ≈ -2.398

score = -1.099 + (-2.398) = -3.497
```

**Word `"meeting"`:**

```
count = word_counts["ham"]["meeting"] = 1
total = 3

log((1 + 1) / (3 + 8)) = log(2 / 11) ≈ -1.705

score = -3.497 + (-1.705) = -5.202
```

**Final ham score: `-5.202`**

### The Winner

| Class | Score |
|-------|-------|
| **spam** | **-4.584** ✅ |
| ham | -5.202 |

`spam` has the higher score → **prediction: `"spam"`**

### Why?

- `"free"` appeared **2 times in spam**, **0 times in ham** → strong spam signal.
- `"meeting"` appeared **0 times in spam**, **1 time in ham** → weak ham signal.
- Spam also had the higher **prior** (2/3 vs 1/3).

The `"free"` signal + prior advantage outweighs the `"meeting"` signal → **spam wins.**

---

## Why Logs?

Without logs, probabilities get multiplied directly:

```
(2/3) × (3/14) × (1/14) ≈ 0.00051   (spam)
(1/3) × (1/11) × (2/11) ≈ 0.00055   (ham)
```

With more words, these become **insanely tiny** and eventually underflow to `0.0` in floating point.

**Logs fix this** because:

```
log(a × b × c) = log(a) + log(b) + log(c)
```

- **Multiplication becomes addition** — numerically safe.
- Log is **monotonic** — the highest log-score is still the highest probability.

That's why `math.log` appears everywhere in `predict()`.

---

## Summary

### The Complete Flow

```
predict("free meeting"):
  1. Split into ["free", "meeting"]
  2. For each class:
       start with log(prior)
       for each word:
           add log(smoothed likelihood)
  3. Return the class with the highest score
```

### Key Formulas

**Prior:**

```
P(class) = class_counts[class] / total_docs
```

**Likelihood (with smoothing):**

```
P(word | class) = (count + smoothing) / (total + smoothing × vocab_size)
```

**Score:**

```
score(class) = log P(class) + Σ log P(word | class)
```

### The Big Picture

| Step | What Happens |
|------|--------------|
| **Train** | Count documents per class, words per class, total words per class, and build vocabulary. |
| **Predict** | For each class: start with log prior, add log likelihood of each word, pick the highest. |
| **Smoothing** | Prevents unseen words from zeroing out predictions. |
| **Logs** | Prevent numerical underflow when multiplying many small probabilities. |

**That's it.** Count → multiply (via logs) → pick the max. No iterations, no gradients — just simple arithmetic that works surprisingly well.