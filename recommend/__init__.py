import logging
import json
import azure.functions as func
from . import ai

ai = ai.AI()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    result = ai.test()

    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json"
    )

