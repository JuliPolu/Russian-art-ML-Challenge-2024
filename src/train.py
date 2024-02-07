import argparse

import torch

from config import Config
from dataset_process import dataset_prepare_split
from model import init_model, calc_embeddings, train_and_save_classifier, load_and_evaluate_classifier


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = arg_parse()
    config = Config.from_yaml(args.config_file)

    x_train, train_labels, x_test, test_labels = dataset_prepare_split(config.data_csv_path, config.data_csv_path)
    
    device = torch.device(config.device)
    model, processor = init_model(config.model_arch, device)

    X_train, y_train = calc_embeddings(model, x_train, train_labels, processor)
    X_test, y_test = calc_embeddings(model, x_test, test_labels, processor)
    
    train_and_save_classifier(X_train, y_train, config.result_model_path, config.scaler_path)
    load_and_evaluate_classifier(config.result_model_path, config.scaler_path, X_test, y_test)
    
    

    