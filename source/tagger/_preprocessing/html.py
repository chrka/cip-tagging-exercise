from html.parser import HTMLParser
from sklearn.base import BaseEstimator, TransformerMixin


class _HTMLToTextParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._text = ''

    def _append_text(self, text):
        self._text += text

    def handle_data(self, text):
        self._append_text(text)

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self._append_text('\n\n')
        elif tag == 'br':
            self._append_text('\n')

    def handle_startendtag(self, tag, attrs):
        if tag == 'br':
            self._append_text('\n')

    def get_text(self):
        return self._text.strip()

def _html_to_text(html):
    parser = _HTMLToTextParser()
    parser.feed(html)
    parser.close()
    return parser.get_text()


class HTMLToText(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(_html_to_text)
