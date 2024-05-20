import os
from argparse import ArgumentParser
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
import PIL
import data
from tensorflow.keras.preprocessing import image
from data import transform

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir-model", default='pokemon_models/checkpoint-1610', type=str)
    parser.add_argument("--checkpoint", default='google/vit-base-patch16-224-in21k', type = str)
    parser.add_argument("--image-path", default='0cfe57a5bf674650b0de0c381df13ca0_jpg.rf.cf29339aa61d57131478f066ba7cceba.jpg', type=str)
    parser.add_argument("--data-dir", default='fcakyon/pokemon-classification', type=str)
    parser.add_argument("--test-size", default=0.2, type=float)

    # FIXME
    args = parser.parse_args()

    # FIXME


    print('---------------------Welcome to Image Classification using Transformer-------------------')
    print('Github: Dungfx15018')
    print('Email: dungtrandinh513@gmail.com')
    print('---------------------------------------------------------------------')
    print('Predict Image Classification model with hyper-params:')
    print('===========================')

    # FIXME
    # Do Training

    dataset = load_dataset(args.data_dir, name="full")
    full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    full_dataset = concatenate_datasets([full_dataset, dataset["test"]])

    shuffled_dataset = full_dataset.shuffle(seed=42)
    splitted_dataset= shuffled_dataset.train_test_split(test_size=args.test_size)

    train_dataset = splitted_dataset['train'].with_transform(transform)
    test_dataset = splitted_dataset['test'].with_transform(transform)

    labels = train_dataset.features['labels'].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = ViTForImageClassification.from_pretrained(args.dir_model)
    image_processor = ViTImageProcessor.from_pretrained(args.checkpoint)



    output = image_processor((PIL.Image.open(args.image_path)), return_tensors='pt')

    with torch.no_grad():
        logits = model(**output).logits

    predictions = logits.argmax(dim=-1).item()
    predict_label= model.config.id2label[predictions]

    x =image.load_img(args.image_path)

    classifier = pipeline('image-classification', args.dir_model)
    print(predict_label)
    print(classifier(x))