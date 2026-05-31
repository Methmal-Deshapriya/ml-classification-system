from pathlib import Path
import pickle
import sys


MODEL_PATH = Path("app/models/bengaluru_house_price_linear_regression_model.pickle")


def inspect_pickle_model(path):
    print("Python version:", sys.version)
    print("Model path:", path.resolve())
    print("File exists:", path.exists())
    print("-" * 60)

    with open(path, "rb") as file:
        loaded_object = pickle.load(file)

    print("Top-level object type:")
    print(type(loaded_object))
    print("-" * 60)

    if isinstance(loaded_object, dict):
        print("This pickle contains a dictionary.")
        print("Dictionary keys:")
        print(list(loaded_object.keys()))
        print("-" * 60)

        for key, value in loaded_object.items():
            print(f"Key: {key}")
            print(f"Type: {type(value)}")

            if hasattr(value, "predict"):
                print("Has predict(): Yes")

            if hasattr(value, "n_features_in_"):
                print("Number of features expected:", value.n_features_in_)

            if hasattr(value, "feature_names_in_"):
                print("Feature names from model:")
                print(list(value.feature_names_in_))

            if isinstance(value, list):
                print("List length:", len(value))
                print("First 20 values:", value[:20])

            print("-" * 60)

    else:
        print("This pickle does NOT contain a dictionary.")
        print("It directly contains an object/model.")
        print()

        model = loaded_object

        if hasattr(model, "predict"):
            print("Has predict(): Yes")

        if hasattr(model, "n_features_in_"):
            print("Number of features expected:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            print("Feature names:")
            print(list(model.feature_names_in_))
        else:
            print("No feature_names_in_ found.")
            print("This model does not directly store feature column names.")

        if hasattr(model, "get_params"):
            print()
            print("Model parameters:")
            print(model.get_params())

        if hasattr(model, "coef_"):
            print()
            print("Coefficient shape:")
            print(model.coef_.shape)

        if hasattr(model, "intercept_"):
            print()
            print("Intercept:")
            print(model.intercept_)


inspect_pickle_model(MODEL_PATH)