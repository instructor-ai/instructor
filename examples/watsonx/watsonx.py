import os

from ibm_watsonx_ai.foundation_models import ModelInference as Watsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import instructor


api_key = os.environ.get("WATSONX_API_KEY")
project_id = os.environ.get("WATSONX_PROJECT_ID")
instance_url = "https://us-south.ml.cloud.ibm.com"

wx = Watsonx(
    model_id="meta-llama/meta-llama-3-70b-instruct",
    credentials={"apikey": api_key, "url": instance_url},
    project_id=project_id,
    params={
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.DECODING_METHOD: "sample",
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1,
    },
)


client = instructor.from_watsonx(wx, mode=instructor.Mode.MD_JSON)
