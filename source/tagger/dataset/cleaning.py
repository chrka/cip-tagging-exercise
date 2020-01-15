import numpy as np
import pandas as pd
import fasttext
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

    # Convert time strings to actual times
    cip_df['time'] = pd.to_datetime(cip_df['time']).dt.time

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
    """Load and split dataset."""
    events_df, tags_df = load_raw_normalized_dataset(path,
                                                     drop_missing=drop_missing)
    top_tags = calculate_top_tags(tags_df, n_tags=n_tags)

    # Only keep top tags
    tags_df = tags_df[tags_df['tag'].isin(top_tags)]

    tag_matrix = tags_to_matrix(events_df, tags_df, top_tags)

    # Split data into public training set and private test set
    # (for exercise, take tag status into account)
    events_train, events_test, tags_train, tags_test = \
        train_test_split(events_df, tag_matrix, test_size=test_size,
                         random_state=random_state)

    return events_train, tags_train, events_test, tags_test, top_tags


def extract_corpus(events_df):
    """Extract text corpus from event descriptions."""
    from tagger._preprocessing.html import HTMLToText
    from tagger._preprocessing.characterset import CharacterSet
    from tagger._preprocessing.lowercase import Lowercase
    from sklearn.pipeline import Pipeline
    cleaning_pipeline = Pipeline([
        ('html', HTMLToText()),
        ('cset', CharacterSet(punctuation=False)),
        ('lcase', Lowercase())
    ])
    return list(cleaning_pipeline.fit_transform(events_df['description']))

def fasttext_wordvectors(corpus_path, model_path):
    model = fasttext.train_unsupervised(corpus_path)
    model.save_model(model_path)
    return model

def save_corpus(events_df, path):
    corpus = extract_corpus(events_df)
    with open(path, 'w') as f:
        for doc in corpus:
            f.write(doc + '\n')

if __name__ == '__main__':
    import os

    print("Current working directory:", os.getcwd())

    events_df, _ = load_raw_normalized_dataset(
        "../../../data/raw/citypolarna_public_events_out.csv",
        drop_missing=True)
    CORPUS_PATH = '../../../data/corpus.txt'
    MODEL_PATH = '../../../data/wordvectors.bin'
    save_corpus(events_df, CORPUS_PATH)
    model = fasttext_wordvectors(CORPUS_PATH, MODEL_PATH)

