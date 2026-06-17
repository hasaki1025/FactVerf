from lora.remote_lora.dataset import create_remote_dataset

if __name__ == '__main__':

    train_dataset_file = '/media/shared_e/lyq/DataSet/REMOTE/datasets/train_preprocessed.json'
    val_dataset_file = '/media/shared_e/lyq/DataSet/REMOTE/datasets/val_preprocessed.json'
    test_dataset_file = '/media/shared_e/lyq/DataSet/REMOTE/datasets/test_preprocessed.json'
    train_data = create_remote_dataset(
        dataset_file=train_dataset_file,
        image_dir='/media/shared_e/lyq/DataSet/REMOTE/datasets/UMKE_IMG',
        tokenizer_name_or_path='/media/shared_d/lyq/models/Qwen3.5-9B',
        max_seq_len=8192,
        cache_dir="/media/shared_e/lyq/DataSet/REMOTE/datasets/cache/train"
    )

    val = create_remote_dataset(
        dataset_file=val_dataset_file,
        image_dir='/media/shared_e/lyq/DataSet/REMOTE/datasets/UMKE_IMG',
        tokenizer_name_or_path='/media/shared_d/lyq/models/Qwen3.5-9B',
        max_seq_len=8192,
        cache_dir="/media/shared_e/lyq/DataSet/REMOTE/datasets/cache/val"
    )

    test = create_remote_dataset(
        dataset_file=test_dataset_file,
        image_dir='/media/shared_e/lyq/DataSet/REMOTE/datasets/UMKE_IMG',
        tokenizer_name_or_path='/media/shared_d/lyq/models/Qwen3.5-9B',
        max_seq_len=8192,
        cache_dir="/media/shared_e/lyq/DataSet/REMOTE/datasets/cache/test"
    )


