import argparse
import json
import wandb
from experiment_config import ExperimentConf
from train_and_evaluate import run_experiment


def run_experiments(config_file):
    """
    runs a set of experiments whose parameters are described in config_file
    @param config_file: json file containing a list containing a set of parameters for each experiment
    """

    with open(config_file,'r') as f:
        json_content = json.loads(f.read())
        for conf in json_content['configs']:
            experiment_conf = ExperimentConf(conf)
            wandb.init(project="test-project", entity="nabil-tyrex", tags=["3 kenn layers", "tyrex-gpu"], config=conf)
            run_experiment(experiment_conf)
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('config', metavar='file', type=str, help='config file in json format')
    args = parser.parse_args()
    run_experiments(args.config)


if __name__ == '__main__':
    main()
