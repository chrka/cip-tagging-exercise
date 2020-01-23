from tagger._evaluation.metrics import exact_match_ratio, hamming_loss
from tagger._evaluation.perlabel import scores_per_label

from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split

def display_stat(name, value):
    display(Markdown(f"**{name}:** {value}"))

def evaluate_classifier(model, top_tags, events, tags, test_size=0.2,
                        random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        events, tags, test_size=test_size, random_state=random_state)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    display_stat("Hamming loss for model", hamming_loss(y_test, y_pred))
    display_stat("Exact match ratio for model",
                  exact_match_ratio(y_test, y_pred))

    return scores_per_label(top_tags, y_test, y_pred).sort_values(
        'f1', ascending=False)