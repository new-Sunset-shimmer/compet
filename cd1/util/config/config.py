from sacred import Experiment

EXPERIMENT_NAME = 'book'

ex = Experiment(EXPERIMENT_NAME, save_git_info=False)

@ex.config
def config():
    exp_name = EXPERIMENT_NAME
    mode = 'train'
    seed = 19
    
    # Model Hyper-Parameter Setting
    num_labels = 24
    # class_weighting = [0.81720045753503, 1.5216986155484558, 1.09282982791587, 2.0529813218390807, 0.8174342105263158, 0.8176680972818312, 0.8169668381932533, 0.8176680972818312, 1.2706758559359714, 0.8174342105263158, 1.423891380169407, 1.550596852957135, 0.8174342105263158, 0.8174342105263158, 0.8179021179164282, 1.6348684210526316, 0.8176680972818312, 1.5858768035516093, 0.8179021179164282, 0.8167333523863961, 0.8167333523863961, 0.8169668381932533, 0.9674170616113744, 2.485]
    class_weighting = []
    # GPU, CPU Environment Setting
    num_nodes = 1
    gpus = [0,1,2,3]
    batch_size = 256
    per_gpu_batch_size = 2   # Note that -> batch_size % (per_gpu_batch_size * len(gpus) == 0
    num_workers = 3

    # Main Setting
    input_seq_len = 128
    image_resolution = 11 #  resolution % 16 == 0
    # num_train_epochs = max_steps / len(train_dataloader)
    # max_steps = 100000
    max_steps = 1500
    warmup_steps = 300
    lr = 2e-5
    # lr = 2e-4, 1e-5, 2e-5, 3e-5, 4e-5
    val_check_interval = 0.2
    model_name = 'bertugmirasyedi/deberta-v3-base-book-classification'
    
    # Path Setting
    load_path = "/home2/yangcw/result/book_seed19_from_/version_6/checkpoints/epoch=2-step=698-val_acc=0.89127.ckpt"
    log_dir = 'result'
    train_dataset_path = r"/home2/yangcw/clear_train_v5.csv"
    val_dataset_path = r"/home2/yangcw/clear_val_v5.csv"
    test_dataset_path = r"/home2/yangcw/clear_test_v4.csv"
    # test_dataset_path = r"/disk/leesm/yangcw/test_data.csv"