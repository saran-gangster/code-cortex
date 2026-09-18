# AeroGuard — Review 1 team script

Target: 6 minutes 30 seconds, including a 65-second demo. Use Speaker 1/2/3 as placeholders; combine parts for a smaller team. Bracketed text is a stage cue, not spoken text. Rehearse at a calm pace. Q&A is extra.

![AeroGuard Review Architecture](assets/aeroguard-review1-architecture.png)

Evidence snapshot: 18 September 2026. ImageNet summaries record 5,000 completed updates per arm. Four development reports are complete. Final-test metrics remain unavailable.

## 0:00–0:55 — Speaker 1: the problem

“Hello. Our project is AeroGuard, a flight-aware aerial detection and review console.

Think of an operator checking drone images of a road. Changes in height, speed and angle change the view. An image-only model does not directly use the flight information recorded with each image.

Our question is simple: does adding the right flight information help the model find objects on recordings it has not trained on?

The operator can see what the model found, which inputs it used, and when an input is missing. This supports human review, not drone control.”

## 0:55–1:40 — Speaker 1: the solution and data

“We compare two versions of the same detector. E1 sees the colour image only. E2 sees the image and paired flight state. Both get the same training budget and image order.

AU-AIR has 32,823 paired frames. Our parser rejected and logged 54 invalid boxes, leaving 131,977 valid boxes across eight classes.

Nearby frames can look almost the same. We keep related streams together: five recording roots for training, one for development and two sealed for final testing.

Speaker 2 will explain how the model uses this data.”

## 1:40–2:50 — Speaker 2: architecture and stack

[Point to the main flow in the diagram.]

“First, we check images, boxes, labels and units. Height is converted to metres. We fit the state scaling only on training data.

Both paths use FCOS with ResNet-50 and a feature pyramid. These find image patterns at several scales. FCOS predicts boxes, classes and scores.

E1 uses height, three speed values, roll, pitch and two yaw values. FiLM scales and shifts image features using this state. A gate switches the adjustment off when state is missing. Invalid values are cleaned first.

Python, PyTorch, TorchVision and Lightning handle training and checkpoints. FastAPI and Pydantic handle requests and input checks. React, TypeScript and Vite build the console. Evidence and reviews use local JSON and JSONL files.”

## 2:50–3:50 — Speaker 2: real engineering evidence

“We have trained models. The saved random-control reports show 5,000 updates for each arm. Both used a pool of 18,523 training records and the same start weights and image schedule. E2 used physical T4 GPU zero; E1 used GPU one. Both used Lightning and full precision.

Checkpoint hashes identify each saved model. Final training losses are about 1.635 and 1.671. They do not tell us which model is more accurate.

ImageNet fine-tuning reports also show 5,000 completed updates per arm. Only the backbone uses ImageNet weights. The AU-AIR head starts random; FiLM starts with no feature change.

Speaker 3 will show how the operator reviews the evidence.”

## 3:50–4:55 — Speaker 3: live console demo

[0–10 seconds: open a prepared recording. Read its visible source badge.]

“First, we check the source label. A fixture is sample data used to show the workflow. Cached output is a saved result. Neither means a model is running now.”

[10–25 seconds: select a frame; show the state panel and comparison control.]

“Here are the image, flight state and model mode. The comparison control lets us inspect the two paths. Only real saved model output can support a performance claim.”

[25–40 seconds: use the missing-state intervention and show the status.]

“Now we remove state in this demo. The screen marks the input as degraded. This is a controlled demonstration, not a measured sensor failure.”

[40–55 seconds: open evidence, add a review and export it.]

“We can inspect the source and model information, record a decision and export the review.”

[55–65 seconds: show results/limitations.]

“The Results page labels these values as non-final development evidence. Synthetic fixtures stay separate and never appear as model accuracy.”

## 4:55–5:50 — Speaker 3: evaluation and limits

“We evaluated all four checkpoints on the same frozen development recording. In the matched ImageNet pair, E2 reached AP50 0.0120 and AP50:95 0.0024. E1 reached 0.0692 and 0.0234. Overall recall rose from 0.333 to 0.493, and human recall rose from 0.014 to 0.061.

This supports E1 on the development recording. It does not prove broad generalization, and the final test is still sealed. The current 0.30 display threshold gives many false positives, so the release threshold and missing-state fallback are not frozen.

VisDrone will be an external image-only test. It has no paired flight state, so we will mark state unavailable. That checks behavior on another dataset.

Eight recording roots give limited variety. We also need more empty-scene checks. This is an offline review prototype, not a proven autonomous safety system.”

## 5:50–6:30 — Speaker 1: progress and closing

“For Review 1, we have data checks, a fixed split, real training records, state gating, development evidence and a review workflow. E1 is the leading checkpoint, but the release model still needs fallback selection and local runtime integration.

Next, we will test E1 with missing state, choose the operating threshold, run the external test, and rehearse the demo offline. At larger scale, the plan adds object storage, a database and queued GPU workers. Those are future work.

AeroGuard makes aerial detections easier to inspect: what the model found, which flight context it used, and what happens when that context is missing. Thank you.”

## Likely judge questions

**Did you train a model or only call an API?** Both the random-control and ImageNet pairs completed 5,000 updates per arm. Checkpoint and schedule hashes are recorded. We also completed four real development evaluations; the API serves those saved reports.

**What is new here?** Our contribution is the controlled use of paired flight state, missing-state handling and a review workflow with traceable evidence. FCOS and FiLM are existing methods that we build on.

**Does flight state improve accuracy?** On the single frozen development recording, yes: the matched ImageNet E2 arm is above image-only E1 on AP and recall. That supports the idea but does not prove generalization. Final-test results are unavailable, and training losses were not used for this conclusion.

**Why FCOS?** Its feature pyramid is accessible, so we can add and test conditioning at a clear point. It already provides detection outputs and losses. We do not claim it is the fastest detector.

**What does FiLM do?** It scales and shifts feature channels using flight state. Its last layers start at zero, so the initial transform changes nothing. Training can learn useful adjustments.

**Is missing-state E1 exactly the same as E2?** No. The zero gate removes E1's state adjustment, but its image weights may have changed during training. We must compare both paths on development data before choosing a fallback.

**How do you avoid leakage?** Related streams stay in one recording-root partition. State scaling uses training roots only. GPS, filenames, dates and recording IDs are excluded from model inputs. The saved runs list no development or final-test roots used for training.

**What does “pretrained” mean here?** The new pair starts its ResNet-50 image backbone from ImageNet weights. The AU-AIR detection head is seeded random; FiLM starts as an identity. This is not a pretrained AU-AIR detector or a claimed COCO detection warm start.

**Why two GPUs and Lightning?** Separate T4s run the two matched arms. Lightning organizes updates, logs and checkpoints. GPU kernels do the heavy maths. We used full precision after the early half-precision check produced invalid gradients; we claim no measured speedup from Lightning alone.

**Which losses and metrics do you use?** The plan keeps FCOS classification, box-regression and centerness losses. The evaluator supports AP50, AP50:95, per-class results, per-root results and fixed-threshold recall. AP50 uses a 50% box-overlap threshold; AP50:95 averages stricter overlap thresholds too.

**Is the demo live inference?** Read the mode badge. The documented console works with fixtures and saved records. A local trained runtime still needs a release checkpoint. We will only describe inference as live when that path is actually connected and verified.

**What database do you use, and how will it scale?** Current reviews use JSONL files; we do not claim a production database. The plan adds PostgreSQL for metadata, object storage for images and queued GPU workers. Capacity and cost still need measurement.

**Why VisDrone? Why only two datasets?** AU-AIR supports the paired-state question. VisDrone checks another image domain with state unavailable. Its seven-concept mapping and ignore policy are custom, not the official ten-class benchmark. The other supplied datasets address different tasks.

**Have you completed 40% or submitted the project?** We show concrete completed parts rather than assign ourselves a score. The rubric asks for at least 40%; judges assess that progress. These sources do not establish a completed submission or receipt.

**Is it safe or ready for commercial use?** This is supervised review software. Safety validation is not established. Dataset rights need review before commercial use or redistribution.

## Evidence for the presenters

Use [random E2](../reports/development_random_e2_summary.json), [random E1](../reports/development_random_e1_summary.json), [ImageNet E2](../reports/development_imagenet_e2_summary.json), [ImageNet E1](../reports/development_imagenet_e1_summary.json), [weight origin](../reports/shared_imagenet_warmstart_summary.json) and [resolved config](../configs/development_training.yaml) for training statements. Use [development results](DEVELOPMENT_RESULTS.md) and its four linked JSON reports for accuracy statements. The [smoke report](../reports/fcos_overfit_gate_summary.json) and [inference record](../reports/fcos_overfit_gate_inference.json) are engineering evidence only. The [synthetic report](../reports/evaluation_pipeline_fixture.json) tests the evaluator only.

Scope, model details and limits come from [README](../README.md), [model card](MODEL_CARD.md), [data card](DATA_CARD.md), [architecture](ARCHITECTURE.md), [development results](DEVELOPMENT_RESULTS.md), and the machine-readable reports. No live status or submission claim follows from those files.
