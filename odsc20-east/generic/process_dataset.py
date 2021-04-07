import pickle
from clearml import Task
from sklearn.model_selection import train_test_split


# Connecting TRAINS
task = Task.init(project_name="ODSC20-east", task_name="process dataset")

# program arguments
args = {
    'dataset_task_id': 'replace_with_data_task_id',
    'random_state': 42,
    'test_size': 0.2,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# get the originating experiment
dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
# download the artifact and open it
iris_pickle = dataset_upload_task.artifacts['dataset'].get_local_copy()
iris = pickle.load(open(iris_pickle, 'rb'))

# "process" data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state'])

# upload processed data
print('Uploading process dataset')
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_test', y_test)

print('Notice, artifacts are uploaded in the background')
