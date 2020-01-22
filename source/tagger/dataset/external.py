import pandas as pd
import nltk

from tagger.dataset.cleaning import save_corpus, fasttext_wordvectors


def load_external(base_url):
    """Load and create external datasets.

    Args:
        base_url: Base URL for datasets

    Returns:

    """
    static_url = base_url + "static/"
    events_train = pd.read_csv(static_url + "events_train.csv")
    events_test = pd.read_csv(static_url + "events_test.csv")
    tags_train = pd.read_csv(static_url + "tags_train.csv").values

    top_tags = list(pd.read_csv(static_url + "top_tags.csv")['tag'].values)
    tags_train_stats = pd.read_csv(static_url + "tags_train_stats.csv")

    # Create wordvectors
    save_corpus(pd.concat([events_train, events_test]), "/tmp/corpus.txt")
    fasttext_wordvectors("/tmp/corpus.txt", "/tmp/wordvectors.bin")

    # Load NLTK stopwords
    nltk.download('stopwords')

    # Setup display defaults for Pandas so that all tags are shown
    pd.set_option('display.max_rows', 72)

    return (events_train, tags_train, events_test, top_tags, tags_train_stats)