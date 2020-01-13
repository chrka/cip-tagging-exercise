import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def load_raw_normalized_dataset(path, drop_missing):
    """Load raw dataset.

    Args:
        path: Path to raw CSV file
        drop_missing: If true, drop events with missing titles or descriptions

    Returns:
        events_df, tags_df: Event and tag dataframes as tuple
    """
    # FIXME: Import 'id' as integer
    cip_df = pd.read_csv(path,
                         header=None,
                         names=['id', 'weekday', 'time', 'title', 'description',
                                'tag_status', 'tag'],
                         na_values=['-01:00:00'])

    # Drop any events with missing titles or descriptions
    cip_df.dropna(subset=['title', 'description'], inplace=True)

    events_df = cip_df.groupby('id').first().drop(
        columns=['tag_status', 'tag']).reset_index()
    tags_df = pd.DataFrame({
        'id': cip_df['id'],
        'tag': cip_df['tag'],
        'verified': cip_df['tag_status'] == 1,
        'removed': cip_df['tag_status'] == 2
    })

    # For sample demo only, ignore verified and remove 'removed' tags
    tags_df = tags_df[~tags_df['removed']]
    tags_df.drop(columns=['verified', 'removed'], inplace=True)

    return events_df, tags_df


def calculate_top_tags(tags_df, n_tags):
    return tags_df['tag'].value_counts().index[:n_tags]


def tags_to_matrix(events_df, tags_df, top_tags):
    """Convert tag table into matrix"""
    # Combine tags into lists
    tags = tags_df.groupby('id')['tag'].agg(lambda x: list(x)).reset_index()

    # Handle events with no top tags
    # TODO: Kludge, write nicer
    missing_tags = pd.DataFrame({
        'id': events_df[~events_df['id'].isin(tags['id'])]['id'].unique()
    })
    missing_tags['tag'] = [[] for _ in range(len(missing_tags))]
    tags = pd.concat([tags, missing_tags])

    # Align tags with events
    aligned_tags = events_df.merge(tags, on='id')

    # Convert aligned tags to matrix
    mlb = MultiLabelBinarizer(classes=top_tags)
    return mlb.fit_transform(aligned_tags['tag'])


def load_datasets(path, drop_missing=True, n_tags=72,
                  test_size=0.2, random_state=42):
    events_df, tags_df = load_raw_normalized_dataset(path,
                                                     drop_missing=drop_missing)
    top_tags = calculate_top_tags(tags_df, n_tags=n_tags)

    # Only keep top tags
    tags_df = tags_df[tags_df['tag'].isin(top_tags)]

    tag_matrix = tags_to_matrix(events_df, tags_df, top_tags)

    # Split data into public training set and private test set
    # (for exercise, take tag status into account)
    events_train, events_test, tags_train, tags_test = \
        train_test_split(events_df,tag_matrix, test_size=test_size,
                         random_state=random_state)

    return events_train, tags_train, events_test, tags_test, top_tags


def extract_corpus(events_df):
    """Extract text corpus from event descriptions."""
    pass


if __name__ == '__main__':
    import os

    print("Current working directory:", os.getcwd())
    events_train, tags_train, events_test, tags_test, top_tags = load_datasets(
        "../../../data/raw/citypolarna_public_events_out.csv")
    print(f"Number of events: {len(events_train)}")
    print(f"Number of tag rows: {len(tags_train)}")

    print(f"Number of missing titles: {events_train['title'].isna().sum()}")
    print(f"Number of missing desc: {events_train['description'].isna().sum()}")

    print(f"Top tags: {top_tags}")

    from tagger.featureextraction.tfidf import Tfidf

    bow = Tfidf()
    docs = [
        ['hej', 'dags', 'igen'],
        ['hej', 'världen', 'hej', 'Världen']
    ]
    print(bow.fit_transform(docs))
    print(bow._vectorizer.vocabulary_)

    from tagger.preprocessing.html import HTMLToText
    from tagger.preprocessing.lowercase import Lowercase
    from tagger.preprocessing.tokenization import Tokenize
    from tagger.classification.naivebayes import NaiveBayes
    from tagger.featureextraction.bags import BagOfWords

    from sklearn.pipeline import Pipeline

    pipeline_bow = Pipeline([
        ('html', HTMLToText()),
        ('lower', Lowercase()),
        ('token', Tokenize()),
        ('bow', BagOfWords()),
        ('clf', NaiveBayes())
    ])

    pipeline_tfidf = Pipeline([
        ('html', HTMLToText()),
        ('lower', Lowercase()),
        ('token', Tokenize()),
        ('tfidf', Tfidf()),
        ('clf', NaiveBayes())
    ])

    pipeline_bow.fit(events_train['description'], tags_train)
    tags_pred_bow = pipeline_bow.predict(events_test['description'])

    pipeline_tfidf.fit(events_train['description'], tags_train)
    tags_pred_tfidf = pipeline_tfidf.predict(events_test['description'])


    print("BoW -----")
    print(tags_pred_bow[0:5])
    print("------")
    print(tags_test[0:5])
    print()

    print("Tfidf -----")
    print(tags_pred_tfidf[0:5])
    print("------")
    print(tags_test[0:5])
    print()


    # Per-row mean accuracy
    print(f"BOW per-row: {np.mean(tags_pred_bow == tags_test, axis=1)}")
    print(f"Tfidf per-row: {np.mean(tags_pred_tfidf == tags_test, axis=1)}")

    # Subset accuracy
    ssa_bow = np.min(tags_pred_bow == tags_test, axis=1)
    print(f"BoW SSA: {np.mean(ssa_bow)}")

    ssa_tfidf = np.min(tags_pred_tfidf == tags_test, axis=1)
    print(f"TFIDF SSA: {np.mean(ssa_tfidf)}")


    print(events_train.iloc[0], tags_train[0])
    print(top_tags[tags_train[0] > 0])