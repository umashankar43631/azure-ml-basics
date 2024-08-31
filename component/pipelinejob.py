from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
# from load_components import loaded_component_prep, loaded_component_train
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

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

@pipeline()
def pipeline_function_name(pipeline_job_input):
    loaded_component_prep = ml_client.components.get('prep_data')
    loaded_component_train = ml_client.components.get('train_model')

    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(input_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.output_data,
    }

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, path="azureml:titanic_data:1"))

print(pipeline_job)

# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"


# Change the default Compute
pipeline_job.settings.default_compute = 'basic-cpu-cluster'
# Change the default Datastore
pipeline_job.settings.default_datastore='workspaceblobstore'
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)