# Monitoring and retraining notes (optional)

- **Drift check:** every week compare the share of predicted BENIGN and the
  median of `Rate`, `Number`, `IAT`, `Header_Length` with the training data.
  A large change means new traffic types. Re-check the model.
- **Refused rows:** if more than a few percent of rows are `refused` or
  `repaired`, the data pipeline has probably changed. Fix the input first.
- **Why drift matters here:** the model was tested on new capture *files*, but
  all files come from one lab. Real networks can look different.
- **Retraining:** collect new labelled files, run `iot-detector train --evaluate`,
  compare `benign_f1` and `log_loss` with the old `.metrics.json`, then tag a new release.
- **Versions:** each saved model stores the library versions. A version warning on
  load means: retrain or install the recorded versions.
