import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

METRICS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score
    # 'auc': roc_auc_score
}


def _per_label_metrics(top_tags, y_true, y_pred):
    stats = []
    for i, tag in enumerate(top_tags):
        stat = {'tag': tag}
        for name, metric in METRICS.items():
            stat[name] = metric(y_true[:, i], y_pred[:, i])
        stats.append(stat)
    return stats


def scores_per_label(top_tags, y_true, y_pred):
    return pd.DataFrame(_per_label_metrics(top_tags, y_true, y_pred))


# TODO: Hide warnings
def evaluate_per_label(model, top_tags, events, tags,
                       test_size=0.2, sample_size=None,
                       n_splits=3, random_state=42):
    # TODO: Support working on a smaller sample of the dataset (sample_size)
    stats = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_index, test_index in tqdm(kf.split(events)):
        X_train, X_test = events.iloc[train_index], events.iloc[test_index]
        y_train, y_test = tags[train_index], tags[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        stats.extend(_per_label_metrics(top_tags, y_test, y_pred))

    return pd.DataFrame(stats).groupby('tag').mean()
