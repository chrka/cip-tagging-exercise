import numpy as np
import pandas as pd
import fasttext
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification, \
    iterative_train_test_split
from functools import reduce

CIP_TAGS = list(map(lambda x: x.strip(),
                    "gratis, mat, musik, kurs, casino, dans, musuem, inlines, "
                    "båt, barn, film, språk, hockey, bowling, fika, sport, "
                    "biljard, bingo, bio, opera, kultur, grilla, kubb, "
                    "festival, cykel, brännboll, picknick, konsert, pub, "
                    "frisbeegolf, mc, gokart, svamp, bangolf, teater, "
                    "afterwork, promenad, humor, utmaning, fest, shopping, "
                    "resa, sällskapsspel, träna, pubquiz, poker, bok, foto, "
                    "hund, skridskor, karaoke, dart, bada, diskussion, "
                    "badminton, pyssel, golf, klättring, loppis, boule, mässa, "
                    "flytthjälp, yoga, innebandy, pingis, handboll, jogga, "
                    "tennis, högtid, astronomi, fiske, beachvolleyboll, "
                    "friluftsliv, volleyboll, geocaching, vindsurfing, "
                    "shuffleboard, SUP, standup, paddel".split(',')))


def load_raw_normalized_dataset(path, drop_missing):
    """Load raw CiP dataset.

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

    # Ignore verified and remove 'removed' tags
    tags_df = tags_df[~tags_df['removed']]
    tags_df.drop(columns=['verified', 'removed'], inplace=True)

    return events_df, tags_df


def calculate_top_tags(tags_df, n_tags, use_cip_tags=True):
    """Calculate top tags from tags dataset

    Args:
        tags_df: Dataset to extract top tags from
        n_tags: Number of topmost tags to get if generating
        use_cip_tags: Use pre-defined tags from CiP (ignores `n_tags`)

    Returns:
        List of topmost tags
    """
    tag_counts = tags_df['tag'].value_counts()
    if use_cip_tags:
        # Not all CiP tags are necessarily present in the dataset
        # and not necessarily in sufficient amounts
        present_tags = set(tag_counts[tag_counts > 5].index)
        return list(filter(lambda t: t in present_tags, CIP_TAGS))
    else:
        return tag_counts.index[:n_tags]


def tags_to_matrix(events_df, tags_df, top_tags):
    """Converts tags to feature matrix

    Args:
        events_df: Events dataset
        tags_df: Tags dataset
        top_tags: Tags to include

    Returns:
        Feature matrix for tags
    """
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


def matrix_to_tags(tags, top_tags):
    top_array = np.array(top_tags)
    joined_tags = []
    for row in tags:
        joined_tags.append(reduce(lambda a, b: a + "," + b, top_array[row > 0]))
    return np.array(joined_tags)


def load_datasets(path, drop_missing=True, n_tags=72,
                  test_size=0.2, random_state=42):
    """Load and split dataset from raw CiP data.

    Args:
        path: Path to raw CiP dataset
        drop_missing: Drop events with no description or title
        n_tags: Number of top tags to use (ignored)
        test_size: Fraction of events to include in test set
        random_state: Random state for the split

    Returns:
        (events_train, tags_train, events_test, tags_test, top_tags,
            tags_train_stats)
    """
    events_df, tags_df = load_raw_normalized_dataset(path,
                                                     drop_missing=drop_missing)
    top_tags = calculate_top_tags(tags_df, n_tags=n_tags)

    # Only keep top tags
    tags_df = tags_df[tags_df['tag'].isin(top_tags)]

    tag_matrix = tags_to_matrix(events_df, tags_df, top_tags)

    # Split data into public training set and private test set
    stratifier = IterativeStratification(
        n_splits=2, order=2,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
        random_state=random_state)
    train_indices, test_indices = next(stratifier.split(events_df, tag_matrix))
    events_train, tags_train = events_df.iloc[train_indices], \
                               tag_matrix[train_indices, :]

    events_test, tags_test = events_df.iloc[test_indices], \
                             tag_matrix[test_indices, :]

    tags_train_stats = pd.DataFrame({
        'tag': top_tags,
        'count': tags_train.sum(axis=0)
    }).sort_values('count', ascending=False)

    return (events_train, tags_train, events_test, tags_test, top_tags,
            tags_train_stats)


def extract_corpus(events_df):
    """Extract text corpus from event descriptions.

    Args:
        events_df: Event dataset

    Returns:
        List of event descriptions as raw text
    """
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
    """Compute word vectors using FastText.

    Args:
        corpus_path: Path to corpus
        model_path: Path for storing FastText model

    Returns:
        FastText model
    """
    model = fasttext.train_unsupervised(corpus_path)
    model.save_model(model_path)
    return model


def save_corpus(events_df, path):
    """Extract and store corpus for events.

    Args:
        events_df: Events dataset
        path: Path for storing corpus
    """
    corpus = extract_corpus(events_df)
    with open(path, 'w') as f:
        for doc in corpus:
            f.write(doc + '\n')


if __name__ == '__main__':
    # Generate static datasets and wordvectors for local dev
    import os

    print("Current working directory:", os.getcwd())

    # Compute word vectors
    events_df, tags_df = load_raw_normalized_dataset(
        "../../../data/raw/citypolarna_public_events_out.csv",
        drop_missing=True)
    CORPUS_PATH = "../../../data/corpus.txt"
    MODEL_PATH = "../../../data/wordvectors.bin"
    save_corpus(events_df, CORPUS_PATH)
    model = fasttext_wordvectors(CORPUS_PATH, MODEL_PATH)

    # Split datasets
    events_train, tags_train, events_test, tags_test, top_tags, tags_train_stats = load_datasets(
        "../../../data/raw/citypolarna_public_events_out.csv"
    )

    print(f"Number of train events: {len(events_train)}")
    print(f"Number of test  events: {len(events_test)}")

    # TODO: Proper path handling
    DATA_PATH = "../../../data/"
    events_train.to_csv(DATA_PATH + "events_train.csv", index=False)
    events_test.to_csv(DATA_PATH + "events_test.csv", index=False)
    # A kludge, but convenient — pandas can load from URL:s
    pd.DataFrame(tags_train).to_csv(DATA_PATH + "tags_train.csv", index=False)
    pd.DataFrame(tags_test).to_csv(DATA_PATH + "tags_test.csv", index=False)

    pd.DataFrame({'tag': top_tags}).to_csv(DATA_PATH + "top_tags.csv",
                                           index=False)
    tags_train_stats.to_csv(DATA_PATH + "tags_train_stats.csv", index=False)