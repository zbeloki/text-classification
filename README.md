# text-classification

Todo:
- Remove binarized y from the dataset and add y(multilabel:bool) instead. Always keep the labels in a 2D matrix and convert on the fly when requesting y(multilabel=False)
- Pass raw labels to sklearn instead of 2D indicator matrix, as it uses multilabel setting otherwise.
- Implement TransformerClassifier
- Don't use/save custom binarizers: sklearn OneVSRestClassifier keep an internal binarizer and transformers use a different binarizer inside config
- Make dataset original format flexible?: sep, text column name, id column name...
- Accept multiple textual columns: title, description, body... (avoid fixed names)