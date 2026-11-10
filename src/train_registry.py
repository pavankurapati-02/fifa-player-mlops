from fifa_churn.experiment import track_experiment


def train_and_register() -> None:
    track_experiment(register=True)


if __name__ == "__main__":
    train_and_register()
