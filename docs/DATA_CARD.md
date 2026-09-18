# AeroGuard data card

## Core dataset: AU-AIR

- 32,823 matched RGB frames at 1920×1080.
- 132,031 raw boxes across Human, Car, Truck, Van, Motorbike, Bicycle, Bus, and Trailer.
- 54 nonpositive boxes are rejected and logged, leaving 131,977 valid boxes.
- Eight recording roots and twelve `_x_`/`_xx_` streams.
- Source labels `0..7` map to model labels `1..8`; model label 0 is background.

The parser preserves literal source aliases `image_width:` and `longtitude`. Altitude is converted from millimetres to metres; velocity stays signed m/s; angles stay radians. `time.ms` is treated as an offset and may exceed 999.

## Frozen protocol

- Train roots: `20190829091111`, `20190905103112`, `20190905111947`, `20190905112522`, `20190905142119`.
- Development root: `20190905091750`.
- Sealed final-test roots: `20190905143505`, `20190906150731`.
- Protocol SHA-256: `c848194b212301de811fc835eb561bfbc6d4ee60bf9f39fb7497c06b97f7a1cc`.

All streams sharing a fourteen-digit recording root remain together. The split was chosen from declared frame/class-support constraints before model scores existed. The normalizer is fitted only on training roots.

## External dataset: VisDrone DET

Reserved for frozen external RGB evaluation using an explicit seven-class mapping and ignore-region policy. It has no paired telemetry; AeroGuard will not infer or invent state for it.

## Deliberately excluded from the core

Dubai segmentation, OpenSky telemetry, and C-MAPSS prognostics represent different targets or modalities without demonstrated sample alignment to AU-AIR. They are documented in the forensic audit but are not mixed into this detector.

## Rights and release

The AU-AIR bundle reports CC BY-NC-SA 2.0 and CC BY-NC 2.0 licenses. Raw data is excluded from the repository. Commercial use, redistribution, and component-specific applicability require primary-source confirmation.
