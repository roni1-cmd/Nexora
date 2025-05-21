import argparse
from src.python.train import train
from src.python.predict import predict

def main():
    parser = argparse.ArgumentParser(description="Nexora ML Project")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        print("Please specify --train or --predict")

if __name__ == "__main__":
    main()
