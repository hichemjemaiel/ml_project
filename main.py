import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    predict,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="ML pipeline CLI")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the saved model"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Predict on the test set"
    )
    parser.add_argument("--load", action="store_true", help="Load the saved model")
    parser.add_argument(
        "--all", action="store_true", help="Run prepare, train and evaluate in order"
    )
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    # --all is a shortcut for --prepare --train --evaluate
    if args.all:
        args.prepare = args.train = args.evaluate = True

    if args.prepare:
        prepare_data()
        print("Data prepared.")

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        x_train, x_test, y_train, y_test = prepare_data()
        predictions = predict(x_test)
        print("First 10 predictions:", predictions[:10])

    if args.load:
        load_model()


if __name__ == "__main__":
    main()
