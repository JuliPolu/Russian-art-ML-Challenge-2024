import argparse
import torch
import joblib
import os

from model import init_model, calc_embeddings
from config import Config


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = arg_parse()
    config = Config.from_yaml(args.config_file)

    #prepare model
    device = torch.device(config.device)
    model, processor = init_model(device, config.model_arch)
    model.eval()

    #prepare test dataset
    dir_list = os.listdir(config.test_path)
    all_image_names = [os.path.join(config.test_path, fname) for fname in dir_list]
    
    #calculate test image embeddings
    X_test = calc_embeddings(model, all_image_names, processor, device)

    #scaling and SVC inference 
    svm_classifier = joblib.load(config.result_model_path)
    scaler = joblib.load(config.scaler_path)
    X_test_scaled = scaler.transform(X_test)
    all_preds = svm_classifier.predict(X_test_scaled)

    #create file for submission
    with open(config.submission_path, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name.split('/')[-1]}\t{cl_id}\n")
            