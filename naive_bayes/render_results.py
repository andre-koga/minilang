# Renders the results of the naive bayes models.

import matplotlib.pyplot as plt
import json
import pandas as pd
import os

def stats_to_df(stats, model):
    df = pd.DataFrame(columns=["model", "language", "word_length", "accuracy"])
    for language in stats:
        for word_length in stats[language]:
            df.loc[df.shape[0]] = [
                model,
                language,
                int(word_length),
                float(stats[language][word_length])
            ]
    return df


paths = [
    '10000_(1, 2, 3)_unweighted_nb',
    '10000_(1, 2)_unweighted_nb',
    '10000_(1,)_unweighted_nb',
    '1000_(1, 2, 3)_unweighted_nb',
    '1000_(1, 2)_unweighted_nb',
    '1000_(1,)_unweighted_nb'
]

dfs = []

for path in paths:
    with open(f"stats/{path}.pkl.json", 'r') as f:
        data = json.load(f)
        df = stats_to_df(data, path)
        dfs.append(df)

df = pd.concat(dfs)

df['word_count'] = df['model'].str.split("_").str[0]
df['ngram_count'] = df['model'].str.count(' ') + 1 # bad but idk how else to do it

accuracy_by_word_length = df.groupby(["model", "word_length"])["accuracy"].mean().reset_index(name="accuracy")

# i'd rather not redo this work but i'm not sure how to do it better
accuracy_by_word_length['word_count'] = accuracy_by_word_length['model'].str.split("_").str[0]
accuracy_by_word_length['ngram_count'] = accuracy_by_word_length['model'].str.count(' ') + 1
accuracy_by_word_length = accuracy_by_word_length.sort_values(by=['word_count', 'ngram_count', 'word_length'])

for word_length in [1, 2, 5]:
    accuracy_by_word_length[accuracy_by_word_length['word_length'] == word_length]\
            .plot(x = "model", y = "accuracy", kind = "bar", title = f"Accuracy by model for word length {word_length}")
    plt.xticks(rotation=15)
    plt.ylim(0, 1)

model_size = {
    p : os.path.getsize(f"models/{p}.pkl") for p in paths
}

accuracy_overall = df.groupby(["model"])["accuracy"].mean().reset_index(name="accuracy")
accuracy_overall['model_size'] = accuracy_overall['model'].map(model_size)
accuracy_overall.plot(x = "model_size", y = "accuracy", kind = "scatter", title = "Accuracy by model size")

plt.show()