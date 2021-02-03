from clearml import Task
from argparse import ArgumentParser

# need to define this to do validation on str to bool
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# parser from zylo117/Yet-Another-EfficientDet-Pytorch
parser = ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
parser.add_argument('--debug', type=boolean_string, default=False,
                    help='whether visualize the predicted boxes of training, '
                         'the output images will be in test/')
parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
parser.add_argument('--log_path', type=str, default='logs/')
parser.add_argument('--saved_path', type=str, default='logs/')

parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
parser.add_argument('--head_only', type=boolean_string, default=False,
                    help='whether finetunes only the regressor and the classifier, '
                         'useful in early stage convergence or small/easy dataset')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                               'suggest using \'admaw\' until the'
                                                               ' very final stage then switch to \'sgd\'')
parser2 = ArgumentParser('more options')
parser2.add_argument('--num_epochs', type=int, default=500)
parser2.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
parser2.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')

parser.add_argument('--es_min_delta', type=float, default=0.0,
                    help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
parser.add_argument('--es_patience', type=int, default=0,
                    help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')

parser3 = ArgumentParser('file options')
parser3.add_argument('-w', '--load_weights', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
parser3.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
parser3.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
parser3.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')

if __name__ == "__main__":
    # this will create "ARGS" in the UI
    task = Task.init(project_name='CLEARML AS GLASS',
                     task_name='too many options 2',
                     auto_connect_arg_parser=False,
                     )

    args = parser.parse_args()
    arg2 = parser2.parse_args()
    arg3 = parser3.parse_args()
    task.connect(args, 'main')
    task.connect(arg2, 'more')
    task.connect(arg3, 'snore')


    ...
    # actual experiment...
    ...
    task.close()
