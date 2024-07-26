# CS4641-language-detection

/*/IO.py: Used for loading and storing different models and different data sources.

/*/Lang.py: This file contains the language code and their respective language name.


/naive_bayes/: Holds all of the files and folders used for the naive_bayes model

/naive_bayes/stats/: Holds all of the json files representing the accuracy of each model according to the language being tested and the amount of words being used.

/naive_bayes/model_testing.py: Used for checking the performance of the model on the test data and storing the results in a file.

/naive_bayes/model_training.py: Used for training the bayes model.

/naive_bayes/render_results.py: Renders the results of the naive bayes models.

/naive_bayes/test_accuracy.py: Tests the accuracy of the model on a given dataset.

/random_forest/: Holds all of the files and folders used for the random forest model

/random_forest/model_training.py: Running this file will train a new random forest model and, once trained, predict the langauge of each inputted word. Parameters are controlled variables

/random_forest/test_accuracy.py: Automatically runs the given model file on a suite of words from each language and saves results to a file

/random_forest/render_results.py: Uses matplotlib to display the result of the accuracy test