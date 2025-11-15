from transformers import pipeline

# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print(sentiment("Infosys shares jump after positive earnings!"))

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
print(ner("Reliance Industries announced a partnership with Adani Power."))


[{'entity_group': 'ORG', 'score': np.float32(0.99963933), 'word': 'Reliance Industries', 'start': 0, 'end': 19}, {'entity_group': 'ORG', 'score': np.float32(0.99876523), 'word': 'Adani Power', 'start': 43, 'end': 54}]