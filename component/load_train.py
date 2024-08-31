from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
# For Creating Environment in AzureML
from azure.ai.ml.entities import Environment
# For Creating COmpute in AzureML
from azure.ai.ml.entities import AmlCompute
# For Component loading
from azure.ai.ml import load_component

# For Machine learning programs
import os
import numpy as np
import pandas as pd


## Load the Component
parent_dir = ''
loaded_component_train = load_component(source=parent_dir + 'train.yml')

# authenticate
credential = DefaultAzureCredential()

SUBSCRIPTION="bce629d2-84e5-4240-b04f-704d6af7377a"
RESOURCE_GROUP="ml-workspaces"
WS_NAME="new-wsp"
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)
    

train = ml_client.components.create_or_update(loaded_component_train)
print("\n\n", train)