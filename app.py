import argparse
import json
from aasist import model as aasist_model
from aasist import trainer as aasist_trainer
from cnn import cnn as cnn_model
from cnn import trainer as cnn_trainer
import tester

def app(args: argparse.Namespace):
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.function == "train_aasist":
        print("Train AASIST model")
        print("Save each best epoch on the disk")
        trainer = aasist_trainer.Trainer()
        trainer.train(config)
    elif args.function == "train_cnn":
        print("Train CNN model")
        print("Save each best epoch on the disk")
        trainer = cnn_trainer.Trainer()
        trainer.train(config)
    elif args.function == "evaluate_sample":
        print("Run evaluation test for a given sample, with a trained model")
        if args.model_path is None:
            raise NotImplementedError("None model path while calling evaluate_sample")
        if args.model_type is None:
            raise NotImplementedError("None model type while calling evaluate_sample")
        if args.protocol is None:
            raise NotImplementedError("None protocol while calling evaluate_sample")
        if args.audio_folder is None:
            raise NotImplementedError("None audio folder while calling evaluate_sample")
        
        model, data_loader = tester.load_model(args.model_type, args.model_path, config)
        tester.evaluate_sample(model, data_loader, args.protocol, args.audio_folder, config)
    elif args.function == "evaluate_signal":
        print("Compute the scores of a random signal from the evaluation set")
        if args.model_path is None:
            raise NotImplementedError("None model path while calling evaluate_sample")
        if args.model_type is None:
            raise NotImplementedError("None model type while calling evaluate_sample")
        
        print(tester.evaluate_signal(args.model_type, args.model_path, args.label, args.attack, config))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti Spoofing System")
    parser.add_argument("-config", dest="config", type=str, help="config file", required=True)
    parser.add_argument("-func_name", dest="function", type=str, help="function to run", required=True)
    parser.add_argument("-model_path", dest="model_path", type=str, help="model path", default=None)
    parser.add_argument("-model_type", dest="model_type", type=str, help="model type", default=None)
    parser.add_argument("-protocol", dest="protocol", type=str, help="data set protocol file", default=None)
    parser.add_argument("-audio_folder", dest="audio_folder", type=str, help="data set audio files folder", default=None)
    parser.add_argument("-label", dest="label", type=str, help="label item to choose", default=None)
    parser.add_argument("-attack", dest="attack", type=str, help="attack type to choose", default=None)
    app(parser.parse_args())
