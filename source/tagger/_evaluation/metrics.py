from sklearn.metrics import accuracy_score, hamming_loss as sklearn_hl

# TODO: Consider adding weight to verified labels

def exact_match_ratio(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def hamming_loss(y_true, y_pred):
    return sklearn_hl(y_true, y_pred)

