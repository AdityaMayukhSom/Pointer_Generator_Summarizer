import os
import pprint
import re

import keras
import tensorflow as tf
from loguru import logger
from rouge import Rouge
from tqdm import tqdm

from batcher import Vocab, batcher
from model import PGN
from test_helper import beam_decode
from training_helper import ModelTrainer


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    logger.info("Building Vocab Object...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    logger.info("Building Batcher Object...")
    dataset_v2 = batcher(params["data_dir"], vocab, params)

    logger.info("Building Optimizer ...")
    optimizer = keras.optimizers.Adagrad(
        learning_rate=params["learning_rate"],
        initial_accumulator_value=params["adagrad_init_acc"],
        clipnorm=params["max_grad_norm"],
    )

    checkpoint_dir = "{}".format(params["checkpoint_dir"])
    model_filepath_format = "pgn_batch_{epoch:05d}.keras"
    checkpoint_filepath_format = checkpoint_dir + model_filepath_format

    def get_latest_checkpoint_number(filenames: list[str]):
        latest_ckpt_number = 1

        for filename in filenames:
            if not filename.startswith("pgn_batch_"):
                continue

            if not filename.endswith(".keras"):
                continue

            regex_search_match = re.search(r"_(\d{5})\.keras", filename)

            if regex_search_match is None:
                continue

            ckpt_number = int(regex_search_match.group(1))
            latest_ckpt_number = max(ckpt_number, latest_ckpt_number)

        return latest_ckpt_number

    logger.info("Building PGN Model ...")

    if not os.path.isdir(checkpoint_dir):
        logger.info("Creating Checkpoint Directory: {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
                
    filenames = os.listdir(checkpoint_dir)

    if len(filenames) > 0:
        latest_batch_number = get_latest_checkpoint_number(filenames)
        checkpoint_filepath = checkpoint_filepath_format.format(epoch=latest_batch_number)
        logger.info("Restoring From {}".format(checkpoint_filepath))
        model = keras.models.load_model(checkpoint_filepath)
        logger.info("Restored From {}".format(checkpoint_filepath))
    else:
        latest_batch_number = 0
        logger.info("Initializing Model From Scratch")
        model = PGN(params, training_mode=True)
        logger.info("Model Initialized From Scratch")

    logger.info("Initializing Model Checkpoint Callback...")
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_format,
        monitor="val_loss",
        save_weights_only=False,
        save_best_only=False,
        mode="auto",
        save_freq=int(params["checkpoints_save_steps"]),
        verbose=1,
    )

    logger.info("Initializing Model Callbacks...")
    callbacks = keras.callbacks.CallbackList(
        [cp_callback],
        add_history=True,
        model=model,
    )

    logger.info("Creating Model Trainer...")
    model_trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        dataset=dataset_v2,
        vocab=vocab,
        batch_size=int(params["batch_size"]),
        maximum_training_steps=int(params["max_steps"]),
        checkpoint_save_steps=int(params["checkpoints_save_steps"]),
    )

    callbacks.on_train_begin()

    logger.info("Starting the training ...")
    model_trainer.execute(callbacks=callbacks, out_file="output.txt", batch_number=latest_batch_number)

    callbacks.on_train_end()


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    logger.info("Building the model ...")
    model = PGN(params, training_mode=False)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(params["data_dir"], vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}".format(params["checkpoint_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=11)

    path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    ckpt.restore(path)
    print("Model restored")

    for batch in b:
        yield beam_decode(model, batch, vocab, params)


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w") as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


def evaluate(params):
    gen = test(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"], position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trial = next(gen)
            reals.append(trial.real_abstract)
            preds.append(trial.abstract)
            pbar.update(1)
    r = Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    pprint.pprint(scores)
