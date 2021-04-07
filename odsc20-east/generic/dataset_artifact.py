from clearml import Task

# create an dataset experiment
task = Task.init(project_name="ODSC20-east", task_name="dataset artifact")

# add and upload local file containing our toy dataset
task.upload_artifact('dataset', artifact_object='iris_dataset.pkl')

print('uploading artifacts in the background')

# we are done
print('see you next time')
