# Model card: TKR segmentation + direct validity/OOD gate

## Summary

This research model accepts an input photograph only when it resembles the project's definition of a valid total knee replacement (TKR) postoperative wound image. For accepted images, the same network also segments the wound and extracts an ROI for a downstream classifier. It is not a YOLO-first pipeline and does not require a separate detector.

The model uses an EfficientNet-B3 encoder, SCSE U-Net decoder, and an image-level validity/OOD head. Its fixed inference size is 256 × 256. The deployed OOD rule is `accept iff ood_score < 0.38371683454513555`; a higher score means more OOD-like.

## Intended use

- Research screening of inputs before TKR wound segmentation.
- Producing a binary wound mask, bounding box, masked image, and ROI.
- Preventing rejected inputs from entering a downstream wound classifier.

This model is not a medical device and must not be used alone for diagnosis, triage, treatment, or patient safety decisions.

## Evaluation snapshot

| Evaluation | Split / groups / images | Result |
|---|---|---|
| Segmentation validation | grouped validation, 8 groups, 32 labeled TKR images | Dice 0.9676; IoU 0.9374; recall 0.9694; precision 0.9664 |
| Gate validation | grouped TKR validation, 54 groups, 349 TKR + 190 OOD | TKR acceptance 100%; non-TKR rejection 100%; AUROC 1.0; AUPRC 1.0; FPR@TPR95 0 |
| Reused comparison set | 46 TKR groups, 245 TKR + 250 OpenImages | TKR acceptance 100%; non-TKR rejection 100%; AUROC 1.0; AUPRC 1.0; FPR@TPR95 0; negative raw/gated non-empty-mask 65/250 → 0/250 |
| Historical SurgWound check | 67 non-TKR images | rejection 100%; negative raw/gated non-empty-mask 48/67 → 0/67 |

The reused comparison and historical checks are retained for comparability; they are not a new locked test. The separate screening of all 2,223 raw TKR images reached 100% acceptance, but it mixes data roles and is only a stress check, not an unbiased test estimate.

## Calibration and selection

The deployment threshold was calibrated using 349 grouped validation TKR images from 54 groups to preserve 100% validation TKR acceptance. No locked-test image was used to choose the threshold. The selected hard-mining checkpoint is epoch 3.

## Important limitations

- “Valid TKR” means similar to the training and validation distribution, not a clinical diagnosis that the image truly depicts TKR surgery.
- Perfect results on the available sets do not prove perfect rejection of every real-world phone screenshot, landscape, unrelated surgery, hospital, camera, lighting condition, crop, blur, or adversarial input.
- Near-OOD surgical wounds and new acquisition sites remain the most important prospective tests.
- Any change to resize/letterbox preprocessing, normalization, model weight, or threshold invalidates the reported calibration.
- `valid_tkr_probability` is derived from the gate score and should not be described as a calibrated clinical probability.
- A passed gate can still produce no wound mask. In that case `should_classify=False` and the downstream classifier must not be called.

## Data and sharing controls

The public/private Git source repository contains code and documentation only. Model weights are distributed privately and verified using the SHA-256 recorded in `manifest.json`. Patient images, mappings, predictions, logs, and checkpoints must not be committed.
