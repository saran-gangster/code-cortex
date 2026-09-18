# AeroGuard Review 2 presentation script

This script follows `AeroGuard_Review2_Evidence_Deck.pptx`. The wording is intentionally simple. One speaker can present it in about six minutes, or the team can divide the slides.

## Slide 1: AeroGuard Review 2 evidence

“Good morning. AeroGuard is a supervised UAV perception and review system. After Review 1, we focused on the questions the judges raised: the exact data split, complete training loss, accuracy-style measures, and clear development results. We also completed the backend so the interface can use real model reports and human reviews.”

## Slide 2: The problem

“Objects look very small from a drone. Their size and viewing angle change when the drone changes height or direction. A normal detector only sees the image. AeroGuard tests whether adding the drone’s flight state helps the detector. The flight values are height, velocity, roll, pitch, and yaw. A human reviews the output. AeroGuard does not control the drone.”

## Slide 3: Dataset split

“AU-AIR contains 32,823 frames. We use 18,523 frames, or 56.43 percent, for training. We use 5,734 frames, or 17.47 percent, for development. The remaining 8,566 frames, or 26.10 percent, form the sealed final test.

We split by complete flight recording. We do not randomly mix video frames. This matters because neighboring frames can look almost identical. Keeping each flight in one split prevents that leakage.”

## Slide 4: Matched model architecture

“Both models use the same FCOS object detector with a ResNet 50 feature pyramid. E1 is the image-only control. Its state gate is zero. E2 receives eight normalized flight values. A small FiLM network uses those values to adjust five feature levels.

Both models started from the same ImageNet backbone and saw the same image order. Flight state was the controlled difference.”

## Slide 5: Complete training record

“Review 1 used 5,000 training samples per model. We have now scaled each model to all 18,523 training frames, which is one complete pass. Both runs used PyTorch Lightning and ran at the same time on separate Tesla T4 GPUs.

We saved the loss at every optimizer step. The faint lines show every value. The strong lines are 200-step moving averages so the trend is easier to read. E1’s average loss moved from 2.2207 in the first 200 steps to 1.3254 in the last 200. E2 moved from 2.2235 to 1.3287.”

## Slide 6: Held-out development results

“We evaluated both models on one complete flight that training never used. E1 reached AP50 0.1698. E2 reached 0.1146. At each model’s best development threshold, E1 reached F1 0.4503 and detection accuracy 0.2906. E2 reached F1 0.2515 and detection accuracy 0.1438.

Detection accuracy here means true positives divided by true positives plus false positives plus false negatives. It is an object-detection measure, not image-classification accuracy.

AP50 checks the full precision and recall curve. A predicted box counts as correct when it overlaps the labelled box by at least 50 percent. Higher is better.”

## Slide 7: Why E2 fell behind

“The full-pass result did not repeat the short Review 1 result where E2 led. We tested the same E2 checkpoint in three ways. Correct state gave AP50 0.1146. Masked state gave 0.1167. State shifted by 1,000 frames gave 0.1138.

These numbers are almost the same. This tells us the main problem was not one wrong state value during inference. Joint training changed the visual detector weights, and those weights generalized poorly to the held-out flight.”

## Slide 8: Ready for the next review

“The FastAPI backend is complete for health checks, capability discovery, reports, inference, replay, and idempotent human reviews. The UI team can work against that stable contract.

We also completed a safer follow-up experiment. E3 and E4 began from the stronger E1 detector. We froze and byte-checked every visual detector tensor, then trained only the small state adapter across all 18,523 training frames. E4 hid state on half of its training steps.

E4 recovered almost all of E1's performance: AP50 was 0.1697 compared with E1's 0.1698, and F1 was 0.4493 compared with 0.4503. It did not beat E1, so we keep E1 as the honest development winner. The final test stays sealed until the model and confidence threshold are frozen.”

## Short answers for likely judge questions

### What are the train and test proportions?

“Training is 56.43 percent, development is 17.47 percent, and the sealed final test is 26.10 percent. The split is by complete flight recording.”

### What is the accuracy?

“At the best development threshold, E1 detection accuracy is 0.2906 and E2 detection accuracy is 0.1438. We define it as TP divided by TP plus FP plus FN. We also report AP50, precision, recall, and F1 because one classification accuracy number is not enough for object detection.”

### Why is the accuracy not close to 100 percent?

“This is a difficult aerial object-detection task. Objects are small, class support is uneven, and the held-out flight differs from training. We report the measured values honestly instead of treating every background pixel as an easy correct answer.”

### Did the loss decrease?

“Yes. The first-to-last 200-step mean moved from 2.2207 to 1.3254 for E1 and from 2.2235 to 1.3287 for E2. The repository includes all 18,523 loss rows for each model.”

### Did flight state help?

“Not reliably. E1 led on the held-out development flight. The frozen-visual E4 follow-up recovered almost all of E1's performance but did not beat it. This confirms that visual-weight drift caused most of E2's drop, while the current flight-state adapter still does not add a measurable development gain.”

### Did you use the final test?

“No. The 8,566-frame final test is still sealed. We will open it once, after the model and threshold are frozen.”
