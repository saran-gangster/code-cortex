# AeroGuard Review 2 — final team presentation script

This script follows `AeroGuard_Review2_Final_Deck.pptx`. The main presentation takes about 7 minutes. Slide 11 is a backup Q&A slide and does not need to be presented unless the judges ask.

## Team split

- **Speaker 1 — problem and progress:** slides 1–3
- **Speaker 2 — data, architecture, and training:** slides 4–6
- **Speaker 3 — results and honest interpretation:** slides 7–8
- **Speaker 4 — working system, demo, and next step:** slides 9–10
- **All speakers:** know the short answers on slide 11

If one person presents, use the same words and ignore the handoff lines.

## Slide 1 — AeroGuard Review 2 final presentation

**Speaker 1 — about 20 seconds**

“Good morning. AeroGuard is a supervised drone-perception review system. It detects small road users in aerial video and lets a human inspect the evidence. For Review 2, we moved from a short pilot to full-pass training, controlled follow-up experiments, a working review console, and a reproducible edge-deployment path. The final test is still sealed.”

## Slide 2 — Problem and solution

**Speaker 1 — about 35 seconds**

“Objects are hard to detect from a moving drone because they occupy very few pixels and their scale changes with height and camera motion. A normal detector uses only the image. AeroGuard also tests whether flight values such as height, velocity, roll, pitch, and yaw can help.

The model produces bounding boxes, object classes, and confidence scores. A human reviews the output. AeroGuard does not control the drone and we are not making an autonomous-safety claim.”

## Slide 3 — What changed after Review 1

**Speaker 1 — about 50 seconds**

“In Review 1, each experiment used a 5,000-update pilot. The early result suggested that the image-plus-flight-state model, E2, might help: its AP50 was 0.0692 compared with 0.0120 for image-only E1.

For Review 2, we trained both arms for all 18,523 training steps and stored every loss value. The conclusion changed. E1 reached AP50 0.1698, while jointly trained E2 reached 0.1146. Stress tests showed that E2’s visual weights had drifted and generalized less well.

We then froze the strong E1 visual detector and trained only the state adapter. E4 recovered to AP50 0.1697, but still did not beat E1. Stronger evidence changed our decision, so E1 remains selected.”

**Handoff:** “I will now hand over to Speaker 2 to explain how we kept the comparison fair.”

## Slide 4 — Dataset and evaluation protocol

**Speaker 2 — about 45 seconds**

“AU-AIR contains 32,823 labelled frames across eight recording roots. We use 18,523 frames, or 56.43 percent, for training. We use 5,734 frames, or 17.47 percent, for development. The remaining 8,566 frames, or 26.10 percent, form the sealed final test.

We split by complete flight recording, not by random video frames. Neighboring frames can look almost identical, so random splitting could leak near-duplicates into evaluation. Keeping each flight in only one split makes the development result more difficult but more honest.”

## Slide 5 — Model architecture and four experiments

**Speaker 2 — about 45 seconds**

“All experiments use the same FCOS detector with a ResNet-50 feature pyramid. E1 is the image-only control. E2 trains the image detector and the flight-state adapter together. The adapter receives eight normalized flight values and uses FiLM to adjust five visual feature levels.

E3 and E4 are corrective experiments. They start from E1 and keep every visual weight frozen while training only the small state adapter. E4 also hides flight state on half of its training steps to make missing telemetry safer.”

## Slide 6 — Complete training evidence

**Speaker 2 — about 45 seconds**

“Each full-pass arm completed 18,523 optimizer steps, which covers the full training split once. E1 and E2 ran in parallel on the two Kaggle T4 GPUs using PyTorch Lightning.

We stored every optimizer-step loss. The faint lines show each step and the strong lines show a 200-step moving average. Both curves fall and then stabilize, so optimization made progress. We still judge model quality on the held-out development flight, not from training loss alone.”

**Handoff:** “Speaker 3 will now explain the measured result and why the numbers are still modest.”

## Slide 7 — Held-out development results

**Speaker 3 — about 55 seconds**

“On the held-out development flight, E1 achieved AP50 0.1698, F1 0.4503, and detection accuracy 0.2906. Jointly trained E2 was weaker at AP50 0.1146. Frozen-visual E3 reached 0.1679, and E4 reached 0.1697.

E4 nearly matched E1, proving that freezing the visual detector corrected the visual-weight drift. However, E4 did not beat E1, so we do not claim that flight state currently improves accuracy. The evidence-based decision is to keep E1. The 8,566-frame final test remains sealed.”

## Slide 8 — Why the scores are modest

**Speaker 3 — about 55 seconds**

“The scores are low, and we are reporting that directly. This is strict object detection on a different complete flight. The evaluation penalizes both false alarms and missed objects; it does not count millions of background pixels as easy correct answers.

At E1’s selected development threshold, there were 6,646 correct detections, 5,996 false alarms, and 10,227 missed objects. This gives 52.57 percent precision, 39.39 percent recall, 45.03 percent F1, and 29.06 percent detection accuracy.

Many aerial targets occupy only a few pixels, the classes are imbalanced, and Review 2 used one full training pass. AP50 0.1698 is the area under a precision–recall curve at 50 percent box overlap. It is not 16.98 percent classification accuracy. These results are not deployment-ready, but they are honest evidence that tells us exactly what to improve.”

**Handoff:** “Speaker 4 will show how this evidence is exposed in the working system and how we plan to deploy it.”

## Slide 9 — Working system and deployment path

**Speaker 4 — about 50 seconds, followed by the demo**

“The FastAPI backend exposes health, reports, replay frames, inference contracts, and human-review records. The React console replays six real AU-AIR development frames with stored GPU-computed E1 and E4 predictions. It supports autoplay, frame seeking, comparison views, evidence inspection, and review export.

For edge deployment, the repository includes ONNX export, TensorRT engine build, output validation, and inference scripts. We do not claim that the TensorRT engine is complete until a frozen checkpoint is exported and benchmarked on the target NVIDIA edge device.”

### 60-second demo sequence

1. Open **Live review** and point to the label `AU-AIR development replay · cached`.
2. Let frames 129–134 advance automatically, then pause on one frame.
3. Switch between **E1**, **E4**, and **Split** to compare stored predictions.
4. Select another frame on the timeline to prove that frame data changes.
5. Open the **Evidence** drawer and show provenance, frame metadata, quality flags, and the review form.
6. Open **Results & evidence** if time permits and point to the full-pass metrics.

Say clearly: “This judge demo is a cached replay of real AU-AIR frames and stored GPU predictions. It is not presented as live TensorRT inference.”

## Slide 10 — Decision and next work

**Speaker 4 — about 35 seconds**

“Review 2 ends with three clear decisions. First, keep E1 because it is the strongest development model. Second, keep the final test sealed until the model, threshold, preprocessing, and deployment contract are frozen. Third, improve the detector before making scaling or safety claims.

Our next steps are longer training, class-balanced sampling, small-object augmentation, validation across more complete flights, and threshold calibration. After that, we will open the final test once and build and benchmark TensorRT on the target edge device.”

**Closing line:** “Our main Review 2 outcome is not an inflated score. It is a stronger baseline, a rejected hypothesis, and a reproducible path to a better final system. Thank you.”

## Slide 11 — likely judge questions

Keep this slide available during Q&A. Use the answers in `JUDGE_QA.md` and do not improvise a stronger claim than the evidence supports.

## Five-minute fallback

If time is cut to five minutes:

- Present slides 1–4 normally.
- On slide 5, explain only E1, E2, and frozen-visual E4.
- On slide 6, say only that all 18,523 steps were stored and both loss trends decreased.
- Present slides 7–10 normally but skip the live demo.
- Keep slide 11 for questions.

## Claims the team must avoid

- Do not call AP50 0.1698 “16.98 percent accuracy.”
- Do not say flight state improved the final selected model.
- Do not call the cached replay live inference.
- Do not call development results final-test results.
- Do not claim a completed TensorRT engine or edge benchmark.
- Do not describe AeroGuard as autonomous flight control or a certified safety system.
