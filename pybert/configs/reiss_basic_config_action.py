from pathlib import Path
BASE_DIR = Path('pybert')
checkpoints_name = "reiss_action_checkpoints" 
config = {
    'raw_data_path': BASE_DIR / 'dataset/train_reiss_all_action.csv',
    'test_path': BASE_DIR / 'dataset/test_reiss_all_action.csv',

    'data_dir': BASE_DIR / 'dataset',
    'data_cache' : BASE_DIR / 'dataset/' / checkpoints_name,
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/" / checkpoints_name,

    'test/checkpoint_dir': BASE_DIR / "output/" / "test/" / checkpoints_name,
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-uncased/bert_vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-uncased',

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased'
}
