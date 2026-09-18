# AeroGuard Review 2 — judge Q&A

## If they ask: “Why is the accuracy so low?”

### Best 30-second answer

“The result is low, and we are not hiding that. This is strict object detection on a different complete flight. Every false alarm and every missed object lowers the score, while millions of background pixels are not counted as easy correct answers. Many targets occupy only a few pixels, the classes are imbalanced, and this run used one full training pass. At the selected threshold, E1 produced 6,646 correct detections, 5,996 false alarms, and missed 10,227 labelled objects. That gives 52.57% precision, 39.39% recall, 45.03% F1, and 29.06% detection accuracy. The model is not deployment-ready; Review 2 gives us an honest baseline and a clear improvement plan.”

### If they challenge the project value

“We agree that the current detector is not ready for a safety claim. The value of Review 2 is that we used a leakage-safe split, exposed the real failure, rejected an unsupported state-fusion claim, and identified visual generalization as the main bottleneck. We now know where to spend the next training budget instead of optimizing against an easy or leaked split.”

### What will improve it?

“We will train for more than one pass, add class-balanced sampling, strengthen small-object and motion-aware augmentation, validate across more complete flights, and calibrate the confidence threshold. We will freeze those choices before opening the final test once.”

## What AP50 means

“AP50 means average precision when a predicted box is accepted only if it overlaps the labelled box by at least 50 percent Intersection over Union. We move the confidence threshold from strict to lenient, measure precision and recall at each point, and calculate the area under that precision–recall curve. Higher is better.”

In raw terms, AP50 rewards a model that finds many objects while keeping false alarms low across many confidence settings. AP50 0.1698 is **not** the same as 16.98% classification accuracy.

## What recall means

“Recall answers: out of all labelled objects, how many did the model find?”

At E1’s selected threshold:

- Correct detections, or true positives: **6,646**
- Missed labelled objects, or false negatives: **10,227**
- Recall: `6,646 / (6,646 + 10,227) = 39.39%`

In simple terms, the model found about 39 of every 100 labelled objects at that operating point.

## What precision means

“Precision answers: out of all boxes predicted by the model, how many were correct?”

At E1’s selected threshold:

- Correct detections: **6,646**
- False alarms: **5,996**
- Precision: `6,646 / (6,646 + 5,996) = 52.57%`

In simple terms, about 53 of every 100 predicted boxes matched a labelled object.

## What detection accuracy means here

We use `TP / (TP + FP + FN)` because ordinary image-classification accuracy is misleading for detection.

- `6,646 / (6,646 + 5,996 + 10,227) = 29.06%`

This measure penalizes both false alarms and missed objects. It does not reward the model for the large amount of empty background in an aerial image.

## Why did Review 1 make E2 look better?

“Review 1 used a shorter 5,000-update pilot. Review 2 used the full 18,523-frame training pass and stronger follow-up tests. The longer run showed that joint image-plus-state training changed the visual detector weights in a way that generalized poorly to the held-out flight. A frozen-visual adapter recovered almost all of E1’s score, but flight state still did not beat E1.”

## Did flight state help?

“Not reliably in the current experiment. E4 with a frozen visual detector reached AP50 0.1697, almost equal to E1 at 0.1698, but it did not exceed it. Therefore E1 remains selected and we do not claim a flight-state gain.”

## What are the train, development, and test proportions?

“Training is 18,523 frames, or 56.43%. Development is 5,734 frames, or 17.47%. The sealed final test is 8,566 frames, or 26.10%. The split is by complete flight recording, not random frames.”

## Why is the final test still sealed?

“The development set is for model and threshold decisions. If we repeatedly inspect the final test, it becomes another development set. We will freeze the model, preprocessing, threshold, and deployment contract, then open the final test once for the final report.”

## Did loss decrease for the entire training run?

“Yes. Every one of the 18,523 optimizer steps per main arm is stored. The 200-step mean loss fell from 2.2207 to 1.3254 for E1 and from 2.2235 to 1.3287 for E2. A lower training loss shows optimization progress, but the held-out development metrics decide which model generalizes better.”

## Is the UI using real data?

“Yes. The demonstration uses six real AU-AIR development frames and stored GPU-computed E1 and E4 predictions. It is intentionally labelled as a cached replay so that we do not confuse it with live inference.”

## Is TensorRT complete?

“The reproducible scripts are complete for ONNX export, TensorRT engine building, validation, and inference. We will claim the engine and its speed only after exporting the frozen checkpoint and benchmarking it on the target NVIDIA edge device.”

## One-sentence metric summary to memorize

“E1 is our current development winner: AP50 0.1698, precision 52.57%, recall 39.39%, F1 45.03%, and detection accuracy 29.06%, measured on a separate complete flight while the final test remains sealed.”
