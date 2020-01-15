from pprint import pprint
from tagger._evaluation.metrics import hamming_loss, exact_match_ratio

def submit_model(model, *, team_name, model_name,
                 local_events=None, local_tags=None):
    if local_events is None:
        raise NotImplementedError("NYI")
    else:
        print(f"Team '{team_name}' submitting model '{model_name}':")
        pprint(model)
        print(72 * '-')

        tags_pred = model.predict(local_events)

        loss = hamming_loss(local_tags, tags_pred)
        print(f"Hamming loss for submission: {loss}")

        emr = exact_match_ratio(local_tags, tags_pred)
        print(f"Exact match ratio for submission: {emr}")