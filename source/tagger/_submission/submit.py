import requests
import pandas as pd
from pprint import pprint
from tagger._evaluation.metrics import hamming_loss, exact_match_ratio

from tagger._evaluation.overall import display_stat

def submit_model(model, events, *, team_name, model_name,
                 local_tags=None,
                 base_url=None):
    print(f"Team '{team_name}' submitting model '{model_name}':")
    pprint(model)
    print(72 * '-')

    tags_pred = model.predict(events)

    if local_tags is None:
        assert base_url is not None
        url = base_url + "submit"
        files = {
            'submission.csv': pd.DataFrame(tags_pred).to_csv(index=False)
        }
        data = {
            'team': team_name,
            'model_description': model_name
        }

        r = requests.post(url, data=data, files=files)

        try:
            r.raise_for_status()
            result = r.json()
            loss = result['hamming_loss']
            emr = result['exact_match_ratio']
        # TODO: Be more specific
        except Exception:
            print("Submission failed!")
            return
    else:
        loss = hamming_loss(local_tags, tags_pred)
        emr = exact_match_ratio(local_tags, tags_pred)

    # TODO: Remove
    display_stat("Hamming loss for submission", loss)
    display_stat("Exact match ratio for submission", emr)