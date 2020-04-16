from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from argparse import ArgumentParser

from absl import app
from absl import flags
from absl import logging

from trains import Task


FLAGS = flags.FLAGS

flags.DEFINE_string('echo_zero', 'zero', 'Text to echo.')
flags.DEFINE_string('another_str', 'My string', 'A string', module_name='test')

task = Task.init(project_name='ODSC20-east', task_name='hyper-parameters example')

flags.DEFINE_integer('echo_one', 1, 'Text to echo.')
flags.DEFINE_string('echo_two', '2', 'Text to echo.', module_name='test')


parameters = {
    'list': [1, 2, 3],
    'dict': {'a': 1, 'b': 2},
    'tuple': (1, 2, 3),
    'int': 3,
    'float': 2.2,
    'string': 'my string',
}
task.connect(parameters)


second_parameters = {
    'value': 2.0,
}

second_parameters = task.connect(second_parameters)

# adding new parameter after connect (will be logged as well)
second_parameters['new_param'] = 'this is new'

# changing the value of a parameter (new value will be stored instead of previous one)
second_parameters['float'] = '9.9'
print(parameters)
print(second_parameters)


def main(_):
    print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info), file=sys.stderr)
    logging.info('echo is %s.', FLAGS.echo_zero)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--argparser_int_value', help='integer value', type=int, default=1)
    parser.add_argument('--argparser_disabled', action='store_true', default=False, help='disables something')
    parser.add_argument('--argparser_str_value', help='string value', default='a string')

    args = parser.parse_args()

    app.run(main)
