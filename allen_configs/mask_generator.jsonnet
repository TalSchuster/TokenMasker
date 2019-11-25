{
    "vocabulary":{
        "directory_path": "vocabulary"
    },
    "dataset_reader": {
        "type": "snli_with_id",
        "token_indexers": {
            "tokens": {
              "type": "single_id",
              "lowercase_tokens": true
            }
        },
    },
    "train_data_path": "https://www.dropbox.com/s/cial78j604u18t1/train.jsonl?dl=1",
    "validation_data_path": "https://www.dropbox.com/s/r7hj9vk1vg8877v/dev.jsonl?dl=1",
    "test_data_path": "https://www.dropbox.com/s/4soqtup0efvxo3m/nli.test.manual_deletions.jsonl?dl=1",
    "model": {
        "type": "mask_generator",
        "classifier_dir" : "https://www.dropbox.com/s/3o2trmtk4n8in9q/neutrality_classifier.tar.gz?dl=1",
        "del_perc_lambda": 0,
        "del_perc": 0,
        "teacher_lambda": 1.0,
        "coverage_lambda": 0.4,
        "transition_lamb": 0.0,
        "gumbel": false,
        "neutral_label": "NOT ENOUGH INFO",
        "bidirectional": true,
        "attention": "bilinear",
        "use_hypothesis": true, 
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
                "embedding_dim": 300,
                "trainable": false
            }
          }
        },
        "projection_size": 100,
        "labeler": {
            "type": "lstm",
            "bidirectional": true,
            "num_layers": 1,
            "dropout": 0.2,
            "input_size": 100,
            "hidden_size": 100
        },
        "contextualizer": {
            "type": "lstm",
            "bidirectional": true,
            "num_layers": 1,
            "dropout": 0.2,
            "input_size": 300,
            "hidden_size": 100
        },
            },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "premise",
                "num_tokens"
            ],
            [
                "hypothesis",
                "num_tokens"
            ]
        ],
        "batch_size": 32
    },
    "evaluate_on_test": true,
    "trainer": {
        "num_epochs": 100,
        "patience": 10,
        "cuda_device": 0,
        "grad_clipping": 5.0,
        "validation_metric": "+acc_vs_del",
        "optimizer": {
            "type": "adagrad"
        }
    }
}
