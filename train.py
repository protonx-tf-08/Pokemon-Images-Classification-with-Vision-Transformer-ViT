import os
import evaluate
from argparse import ArgumentParser
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer, AutoModelForImageClassification, AutoImageProcessor
from data import transform, collate_fn, image_processor, checkpoint


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Add your command-line arguments here
    parser.add_argument("--batch-size", default=64, type=int)
    # parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--num-train-epochs", default=10, type=int)

    args = parser.parse_args()

    print('---------------------Welcome to Pokemon Images Classification with Vision Transformers-------------------')
    print('Github: Dungfx15018 and EveTLynn')
    print('Email: dungtrandinh513@gmail.com and linhtong1201@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training Image Classification model with hyper-params:')
    print('===========================')

    # Process data
    dataset = load_dataset("fcakyon/pokemon-classification", name="full")
    full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    full_dataset = concatenate_datasets([full_dataset, dataset["test"]])

    shuffled_dataset = full_dataset.shuffle(seed=42)
    splitted_dataset= shuffled_dataset.train_test_split(test_size=0.2)

    train_dataset = splitted_dataset['train'].with_transform(transform)
    test_dataset = splitted_dataset['test'].with_transform(transform)

    labels = train_dataset.features['labels'].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Instantiate the model
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Compile the model
    training_args = TrainingArguments(
        output_dir="pokemon_models",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        max_steps=2000,
        num_train_epochs=args.num_train_epochs,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=2e-8,
        warmup_ratio=0.1,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to='tensorboard',
    )
    
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset ,
        tokenizer=image_processor,
    )

    # Evaluate the model
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
   
    metrics = trainer.evaluate(dataset["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
