from pm4py.objects.log.log import EventLog

from src.log.common import get_log
from src.encoding.time_encoding import is_special_occasion
from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes


def create_sintetic_log(CONF=None):
    if CONF is None:
        CONF = {  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': '../input_data/' + 'd1_train_explainability_0-38.xes',   #0-60
                    'VALIDATE_DATA': '../input_data/' + 'd1_validation_explainability_38-40.xes',   #60-80
                    'FEEDBACK_DATA': '../input_data/' + 'd1_test_explainability_40-50.xes',   #60-80
                    'TEST_DATA': '../input_data/' + 'd1_test2_explainability_50-60.xes',   #80-100
                    'OUTPUT_DATA': '../output_data',
                },
        }

    train_log = get_log(filepath=CONF['data']['TRAIN_DATA'])
    validate_log = get_log(filepath=CONF['data']['VALIDATE_DATA'])
    test_log = get_log(filepath=CONF['data']['TEST_DATA'])

    train_log_formatted = format_label_log(train_log)
    validate_log_formatted = format_label_log(validate_log)
    test_log_formatted=format_label_log(test_log)

    return train_log_formatted, validate_log_formatted, test_log_formatted


def format_label_log(log: EventLog):
    for trace in log:
        counter_true = 0
        counter_false = 0
        for event in trace:
            if is_special_occasion(event['time:timestamp']):
                counter_true += 1
            else:
                counter_false += 1

        if counter_true > counter_false:
            trace.attributes['label'] = 'true'
        else:
            trace.attributes['label'] = 'false'

    return log


if __name__ == '__main__':
    train_log_formatted, validate_log_formatted, test_log_formatted = create_sintetic_log(
        CONF={  # This contains the configuration for the run
            'data':
                {
                    'TRAIN_DATA': '../input_data/0-60.xes',
                    'VALIDATE_DATA': '../input_data/60-80.xes',
                    'FEEDBACK_DATA': '../input_data/60-80.xes',
                    'TEST_DATA': '../input_data/80-100.xes',
                    'OUTPUT_DATA': '../output_data',
                },
        }
    )
