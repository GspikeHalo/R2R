from itertools import islice
import tqdm

from uvcgan2.config      import Args
from uvcgan2.data        import construct_data_loaders
from uvcgan2.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan2.cgan        import construct_model
from uvcgan2.utils.log   import setup_logging

from .metrics   import LossMetrics
from .callbacks import TrainingHistory
from .transfer  import transfer
import wandb

def training_epoch(it_train, model, title, steps_per_epoch, start_step):
    model.train()

    steps = len(it_train)
    if steps_per_epoch is not None:
        steps = min(steps, steps_per_epoch)

    progbar = tqdm.tqdm(desc = title, total = steps, dynamic_ncols = True)
    metrics = LossMetrics()
    global_step = start_step

    for batch in islice(it_train, steps):
        model.set_input(batch)
        model.optimization_step()

        current = model.get_current_losses()
        metrics.update(current)

        global_step += 1
        wandb.log(current, step=global_step)

        progbar.set_postfix(metrics.values, refresh=False)
        progbar.update()

    progbar.close()
    return metrics, global_step

def try_continue_training(args, model):
    history = TrainingHistory(args.savedir)

    start_epoch = model.find_last_checkpoint_epoch()
    model.load(start_epoch)

    if start_epoch > 0:
        history.load()

    start_epoch = max(start_epoch, 0)

    return (start_epoch, history)

def train(args_dict):
    args = Args.from_args_dict(**args_dict)

    wandb.init(
        project="StarGAN-R2R",
        entity='bias-lab',
        config=args_dict,
        name='UCVGan V2'
    )

    setup_logging(args.log_level)
    seed_everything(args.config.seed)

    device   = get_torch_device_smart()
    it_train = construct_data_loaders(
        args.config.data, args.config.batch_size, split = 'train'
    )

    print("Starting training...")
    print(args.config.to_json(indent = 4))

    model = construct_model(
        args.savedir, args.config, is_train = True, device = device
    )
    start_epoch, history = try_continue_training(args, model)

    if (start_epoch == 0) and (args.transfer is not None):
        transfer(model, args.transfer)

    global_step = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        title   = 'Epoch %d / %d' % (epoch, args.epochs)
        metrics, global_step = training_epoch(
            it_train, model, title, args.config.steps_per_epoch, global_step
        )

        history.end_epoch(epoch, metrics)
        model.end_epoch(epoch)

        if epoch % args.checkpoint == 0:
            model.save(epoch)

    model.save(epoch = None)
    wandb.finish()
