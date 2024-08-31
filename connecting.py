'''
    How to do entire Machine learning program running on 
    Azure Machine Learning Studio

    1. Connect to Azure Machine Learning Workspace ( You can also use ARM Template to Create)
    2. Create the Environment Inside the Azure Machine Leanring Studio
    3. Create the Compute inside for Running ML Algorithms inside the Machine Learning Studio
'''

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
# For Creating Environment in AzureML
from azure.ai.ml.entities import Environment
# For Creating COmpute in AzureML
from azure.ai.ml.entities import AmlCompute

# For Machine learning programs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# authenticate
credential = DefaultAzureCredential()

SUBSCRIPTIONID=""
RESOURCE_GROUP="azure-learning"
WS_NAME="azureml-sha1"
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTIONID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)
    
# -----******------                               ---******------------
# ------------***** ENVIRONMENT CREATION (Custom) *****----------------
# -----******------                               ---******------------
custom_env_name = "iris-data-env"
dependencies_dir = "./dependencies"

# DELETING an Existing Environment If needed, but this environment can also be restored.
# environment = ml_client.environments.get(custom_env_name,3.0)
# if environment:
#     ml_client.environments.archive(custom_env_name, 3.0)
    # environment.archive()
# Create Environment inside the Azure ML Studio
pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for iris Defaults pipeline",
    tags={"scikit-learn": "latest"},
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="4.0",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}")
# ----------------------                 -----------------------------

# -----**********--------                 -------********------------
## -------------------****COMPUTE CREATION****-----------------------
# -----**********--------                 -------********------------
# Create a compute to run the machine learning learning model on environment
# environment is like os
# compute is like cpu, which runs the programs inside the os

# Name assigned to the compute cluster
cpu_compute_target = "basic-cpu-cluster"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster).result()

print(
    f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}"
)

