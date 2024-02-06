import pandas as pd
import config as config
from maincode.trainer import ModelTrainer
from maincode.dataloader import AudioDataLoader
from models.dense import DenseClassifier
from maincode.function import set_seeds, export_hyperparams


if __name__ == "__main__":
    set_seeds(config.SEED)
    # initialize the model
    dense_model = DenseClassifier(input_shape=config.MODELS.get(config.MODEL_NAME).get("in_dim"),
    num_classes=config.N_CLASSES,dropout=config.DROPOUT,l2_regularization=config.L2_REGULARIZATION,)
    # initialize the trainer
    trainer = ModelTrainer(
        dense_model, config.LEARNING_RATE, export_path=config.EXPORT_PATH
    )
    # compile the model
    trainer.compile()
    # get train, validation and test splits
    train_split = pd.read_csv(
        f"{config.PARTITIONS_PATH}\\split01_train.csv", header=None, names=["filename"]
    )
    ##train/valid split
    train_split=train_split.sample(frac=0.7)
    valid_split=train_split.drop(train_split.index)
    # prepare dataloaders
    trainloader = AudioDataLoader(
        train_split["filename"].tolist(),
        batch_size=config.BATCH_SIZE,
        dataset="train",
        shuffle=True,
    )
    # generate train and test datasets
    train_dataset = trainloader.create_dataset()
    # start training
    trainer.train(train_dataset, epochs=config.EPOCHS)
    # save the model
    trainer.save_model()
    # save the hyperparameters
    export_hyperparams(trainer.runid, f"{trainer.export_path}\\hyperparams.json")
