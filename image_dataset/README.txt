PA04 IMAGE CLASSIFICATION DATASETS

1. TEST SET: 01_TEST_DO_NOT_TRAIN
   AI-generated: 80 images; Real: 80 images.
   Never upload these images for training. Use this same test set for all models.

2. TRAINING SETS: 02_TRAIN
   Size_Small:    AI-generated 50,  Real 50
   Balanced_Base: AI-generated 100, Real 100
   Size_Large:    AI-generated 200, Real 200
   AI_Majority:   AI-generated 160, Real 40
   Real_Majority: AI-generated 40,  Real 160

   Create a separate Teachable Machine image project for each condition.
   Create classes AI-generated and Real. Upload images from the corresponding
   class folder only. Keep Epochs, Batch size and Learning rate fixed.
   Balanced_Base is shared by the data-size and class-imbalance comparisons.
   Overlap across training conditions is intentional: smaller class subsets
   are nested inside larger subsets. Do not combine the condition folders.
   Counts refer to uploaded images, before any internal validation split.

3. UNUSED RESERVE: 03_UNUSED_RESERVE
   AI-generated: 130; Real: 116. Not required for the planned experiments.

4. PREPARATION
   Original counts: AI-generated 411; Real 396.
   Removed one exact duplicate: AiArtData/image34.jpg
   (same source bytes as AiArtData/image33.jpg).
   Unique retained counts: AI-generated 410; Real 396.
   Source files were sorted, then shuffled using Python random.Random:
   AI-generated seed 2039; Real seed 2040.
   First 80 per class are test images; subsequent images form the train pool.
   All files were converted to RGB JPEG (quality 95), EXIF orientation applied,
   transparency composited on white; no resize, crop, or augmentation applied.
   Original names and source SHA-256 hashes are recorded in manifest.csv.
   Class labels follow the supplied archives; provenance was not independently
   verified. Exact source duplicates were checked across both archives.
   Near-duplicates and semantic similarities were NOT checked.
   Before final training, visually check for near-duplicates crossing the test
   and training sets. If any are found, revise the split before any evaluation.

5. RESEARCH QUESTIONS
   RQ1: Compare Size_Small, Balanced_Base, Size_Large.
   RQ2: Compare Balanced_Base, AI_Majority, Real_Majority.
   Evaluate all models on the same 160 test images.
