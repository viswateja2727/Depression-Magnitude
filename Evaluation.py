from sklearn.metrics import precision_score, recall_score, f1_score

# For LSTM predictions
y_pred = (model.predict(X) > 0.5).astype("int32")
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# For BERT predictions
y_pred_bert = tf.sigmoid(bert_model.predict([test_encodings['input_ids'], test_encodings['attention_mask']]).logits)
y_pred_bert = (y_pred_bert > 0.5).numpy().astype(int)
precision_bert = precision_score(y_test, y_pred_bert)
recall_bert = recall_score(y_test, y_pred_bert)
f1_bert = f1_score(y_test, y_pred_bert)
print(f"BERT Precision: {precision_bert:.2f}, Recall: {recall_bert:.2f}, F1-score: {f1_bert:.2f}")
