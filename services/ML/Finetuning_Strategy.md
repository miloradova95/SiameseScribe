# SiameseScribe Finetuning Feedback Strategy

## Original Problem

User feedback is collected as pairwise judgements from the retrieval interface:

```text
(anchor_patch, result_patch, is_similar)
```

Triplet loss requires a full structure:

```text
(anchor, positive, negative)
```

This creates a sparsity problem. A user may only mark one retrieved patch for an anchor, which gives either a positive pair or a negative pair, but not a complete triplet. Waiting until the same anchor receives both feedback types would waste useful feedback and delay finetuning.

Simple fallback options are not ideal. Artist or rubricator groups only work for known training data, not for newly uploaded images. Using the model's top or worst retrieval results can reinforce its current bias. Random negatives may be too easy or noisy. The strategy below keeps triplet loss as the main objective while handling incomplete feedback conservatively.

---

## Goal

Finetuning should use sparse expert feedback without inventing unreliable training samples. The model stays triplet-based wherever possible. Pairwise loss is only used when no safe triplet can be built.

Feedback is grouped by anchor:

```text
anchor A:
    positives = [P1, P2, ...]
    negatives = [N1, N2, ...]
```

---

## Training Sample Types

### 1. Complete User Triplets

If an anchor has at least one confirmed positive and one confirmed negative:

```text
anchor   = A
positive = user-confirmed similar patch P
negative = user-confirmed dissimilar patch N
```

This is the strongest feedback signal.

```text
w_real_triplet = 1.0
```

### 2. Negative-Only Feedback

If an anchor only has negative feedback, create the missing positive by augmenting the anchor:

```text
anchor   = A
positive = augmented(A)
negative = user-confirmed dissimilar patch N
```

This is safer than choosing an uncertain positive from the dataset.

```text
w_augmented_triplet = 0.7
```

### 3. Positive-Only Feedback

If an anchor only has positive feedback, the missing negative is uncertain:

```text
anchor   = A
positive = user-confirmed similar patch P
negative = ?
```

Instead of inventing a random or model-mined negative, use this as an auxiliary positive pair:

```text
(anchor A, positive P) should move closer together
```

```text
alpha_pair = 0.2
```

This keeps triplet loss as the main objective and uses pairwise loss only where triplet generation would be unreliable.

---

## Synthetic Positive Augmentation Policy

The synthetic positive should still clearly represent the same pen flourish. Augmentations should therefore be mild and preserve stroke structure, orientation, and local morphology.

Recommended augmentations for `augmented(A)`:

```text
small crop/resize:      scale=(0.92, 1.00), ratio=(0.95, 1.05)
small rotation:         ±5 degrees
small translation:      up to ±3% of patch size
brightness adjustment:  ±10%
contrast adjustment:    ±10%
```

These five transformations should be enough for a first implementation. They create a slightly different view of the same patch without strongly changing the actual flourish shape.

Avoid stronger transformations:

```text
horizontal or vertical flips
large rotations
large crops
strong perspective transforms
heavy blur
random erasing or cutout over strokes
strong hue changes
```

Reasoning: contrastive learning often uses augmented views of the same image as positives. For manuscript patches, this idea is useful, but the augmentations need to be weaker because shape, stroke direction, and local pen movement are part of the similarity signal.

If the augmented positive looks visibly like a different patch, the augmentation is too strong.

---

## Main Risk

Positive-only pairwise feedback only pulls embeddings closer. If too much of the update comes from positive-only pairs, the embedding space can become less discriminative. Therefore, positive-only pairs should not block finetuning, but they should be capped when they dominate.

Negative supervision comes from:

```text
complete user triplets
negative-only triplets with augmented anchor positives
```

---

## Balance Estimate

For each finetuning run, count:

```text
T_real = number of complete user triplets
T_aug  = number of negative-only augmented triplets
P_pos  = number of positive-only pairwise samples
```

Triplets contain both a positive and a negative signal. Positive-only pairs add only a positive signal.

```text
effective_negative = T_real + (w_augmented_triplet * T_aug)
effective_positive = effective_negative + (alpha_pair * P_pos)
positive_to_negative_ratio = effective_positive / effective_negative
```

Use a relaxed early-development limit:

```text
positive_to_negative_ratio <= 2.0
```

---

## Practical Finetuning Trigger

To avoid delaying finetuning too much, use a permissive trigger rule:

```text
run finetuning if T_real + T_aug >= 1
```

This means a run can start as soon as there is at least one real user triplet or one negative-only item that can be turned into an augmented triplet.

If no negative signal exists:

```text
T_real + T_aug == 0
```

then skip finetuning for now. Do not train only on positive-only pairs.

---

## Capping Positive-Only Pairs

If there are too many positive-only pairs, only include as many as the ratio allows:

```text
allowed_pair_contribution = (MAX_POS_NEG_RATIO - 1.0) * effective_negative
max_positive_pairs = allowed_pair_contribution / alpha_pair
```

Positive-only pairs above this limit are kept in the feedback database but not used in the current finetuning run. They can still be used later if more negative feedback becomes available.

---

## Recommended First Implementation

```text
1. Build complete user triplets where possible.
2. Build augmented-anchor triplets from negative-only feedback.
3. Build positive-only pairwise samples.
4. Run finetuning if at least one real or augmented triplet exists.
5. Cap positive-only pairs if they exceed the balance limit.
6. If no triplet exists, skip finetuning for now.
```

Main loss:

```text
L_total = L_triplet + alpha_pair * L_pairwise_positive
```

---

## Why This Strategy Is Defensible

This keeps the method triplet-based while adapting it to sparse real feedback. It avoids discarding incomplete feedback, avoids inventing unreliable negatives for positive-only feedback, and avoids training only on positive pull signals.

The system can finetune early as soon as at least one real or augmented triplet exists. Missing data is handled by capping positive-only pairs rather than by waiting for a perfectly balanced feedback graph.

---

## Initial Values

```text
w_augmented_triplet = 0.7
alpha_pair = 0.2
MAX_POS_NEG_RATIO = 2.0
minimum trigger = at least one real or augmented triplet
```
