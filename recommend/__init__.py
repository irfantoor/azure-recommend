import logging
import json
import azure.functions as func
from . import ai

ai = ai.AI()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # get params
    type = req.params.get('type')
    key = req.params.get('key')
    item_id = req.params.get('item_id')
    group_id = req.params.get('group_id')
    user_id = req.params.get('user_id')
    
    if type == 'summary':
        result = ai.summary(key)

    elif type == 'details':
        result = ai.details(item_id=item_id, group_id=group_id, user_id=user_id)

    elif type == 'popular_items':
        result = ai.popular_items(group_id=group_id)

    elif type == 'random_items':
        result = ai.random_items(group_id=group_id)

    elif type == 'recommended_items':
        result = ai.recommended_items(item_id=item_id, user_id=user_id)

    elif type == 'recent_items':
        result = ai.recent_items(group_id=group_id, user_id=user_id)

    elif type == 'popular_groups':
        result = ai.popular_groups()

    elif type == 'random_groups':
        result = ai.random_groups()

    elif type == 'test':
        result = ai.test()

    else:
        result = []

    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json"
    )

