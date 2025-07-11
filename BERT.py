from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_bert(texts, tokenizer, max_len=100):
    return tokenizer(
        list(texts), 
        max_length=max_len, 
        truncation=True, 
        padding='max_length', 
        return_tensors='tf'
    )

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2)
train_encodings = encode_bert(X_train, bert_tokenizer)
test_encodings = encode_bert(X_test, bert_tokenizer)

bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])

bert_model.fit(
    [train_encodings['input_ids'], train_encodings['attention_mask']],
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.1
)
