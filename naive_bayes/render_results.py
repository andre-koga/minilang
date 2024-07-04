import matplotlib.pyplot as plt
import json
import os

models = [
    (10000, (1, 2, 3), False),
    (10000, (1, 2), False),
    (10000, (1), False),
    (1000, (1, 2, 3), False),
    (1000, (1, 2), False),
    (1000, (1), False),
]

paths = []
names = []
for model in models:
    tuple_str = str(model[1]) if type(model[1]) == tuple else f"({model[1]},)"
    paths.append(f"{model[0]}_{tuple_str}_{'weighted' if model[2] else 'unweighted'}_nb")
    names.append(f"{model[0]} words, {tuple_str} ngrams")

stats = []
for path in paths:
    with open(f"stats/{path}.pkl.json", 'r') as f:
        stats.append(json.load(f))

accuracy_by_num_words = {}
for i, stat in enumerate(stats): # for each model
    for lang in stat:            # for each language
        for ngram in stat[lang]: # for each ngram
            accuracy_by_num_words[ngram] = accuracy_by_num_words.get(ngram, {})
            accuracy_by_num_words[ngram][names[i]] = accuracy_by_num_words[ngram].get(names[i], 0) \
                                                    + (stat[lang][ngram])

total = len(stats[0].keys())
for num_words in accuracy_by_num_words:
    for model in accuracy_by_num_words[num_words]:
        accuracy_by_num_words[num_words][model] /= total

for num_words in accuracy_by_num_words:
    xs = names
    ys = []
    for model in accuracy_by_num_words[num_words]:
        ys.append(accuracy_by_num_words[num_words][model])

    plt.figure()
    plt.bar(xs, ys)
    plt.title(f"Accuracy by n-grams for {num_words} words")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # 45 degree labels
    plt.xticks(rotation=15)

file_sizes = [os.path.getsize(f"models/{path}.pkl") for path in paths]
plt.figure()
plt.bar(paths, file_sizes)
plt.title(f"Storage Size")
plt.xlabel("Model")
plt.ylabel("MB")
plt.xticks(rotation=15)

overall_accuracy = {}
for num_words in accuracy_by_num_words:
    for model in accuracy_by_num_words[num_words]:
        overall_accuracy[model] = overall_accuracy.get(model, 0) + accuracy_by_num_words[num_words][model]

accuracies = [overall_accuracy[model] / 3 for model in names]
plt.figure()
plt.scatter(file_sizes, accuracies)
plt.title(f"Storage Size")
plt.xlabel("File Size")
plt.ylabel("Accuracy")
for i, txt in enumerate(names):
    plt.annotate(txt, (file_sizes[i], accuracies[i]))

plt.show()