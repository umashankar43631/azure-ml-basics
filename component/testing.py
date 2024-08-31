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
import matplotlib.pyplot as plt
import seaborn as sns

## Load the Component
parent_dir = ''
loaded_component_prep = load_component(source=parent_dir + 'prep.yml')
