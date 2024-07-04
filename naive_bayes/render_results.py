import matplotlib.pyplot as plt
import json
import pandas as pd
import os

def stats_to_df(stats : dict, model : str):
    """
    Convert a stats json to a pandas dataframe

    Parameters:
    - stats: dictionary of the from: {language: {word_length: accuracy, ...}, ...}
    - model: name of the model used to generate the stats

    Returns: pandas dataframe with columns: model, language, word_length, accuracy
    """
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

def main(models, word_lengths):
    """
    Graphs the key stats for the given models

    Parameters:
    - models: list of model paths
    - word_lengths: list of word lengths to graph

    Returns: None
    """
    dfs = []

    for path in models:
        # assumes the stats are in the stats folder with extension .pkl.json
        with open(f"stats/{path}.pkl.json", 'r') as f:
            data = json.load(f)
            df = stats_to_df(data, path)
            dfs.append(df)

    df = pd.concat(dfs)

    df['word_count'] = df['model'].str.split("_").str[0]
    df['ngram_count'] = df['model'].str.count(' ') + 1 # bad but idk how else to do it

    # aggregate stats by word length (so ignoring language) then sort from least to most complex
    accuracy_by_word_length = df.groupby(["model", "word_length"]).agg({
        'accuracy': 'mean',
        'word_count': 'first',
        'ngram_count': 'first'
    }).reset_index().sort_values(by=['word_count', 'ngram_count', 'word_length'])

    for word_length in word_lengths:
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

if __name__ == "__main__":
    models = [
        '10000_(1, 2, 3)_unweighted_nb',
        '10000_(1, 2)_unweighted_nb',
        '10000_(1,)_unweighted_nb',
        '1000_(1, 2, 3)_unweighted_nb',
        '1000_(1, 2)_unweighted_nb',
        '1000_(1,)_unweighted_nb'
    ]
    word_lengths = [1, 2, 5]
    main(models, word_lengths)