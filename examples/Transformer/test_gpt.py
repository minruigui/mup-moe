import torch
from torch.utils.data import Dataset, DataLoader
import random
from .gpt import GPT, GPTConfig
from datetime import datetime
import time
import functools
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from mup.coord_check import _record_coords
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes
from mup.shape import _extract_shapes,_zip_infshape_dict
import os
import math
from mup.optim import dMuAdam as Adam
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
class SortingDataset(Dataset):
    def __init__(self, num_samples=10000, array_size=10, min_value=0, max_value=9):
        self.data = []
        for _ in range(num_samples):
            unsorted_array = [random.randint(min_value, max_value) for _ in range(array_size)]
            sorted_array = sorted(unsorted_array)
            self.data.append((unsorted_array, sorted_array))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unsorted_array, sorted_array = self.data[idx]
        return {
            'input': torch.tensor(unsorted_array, dtype=torch.long),
            'label': torch.tensor(sorted_array, dtype=torch.long)
        }
def cleanup():
    dist.destroy_process_group()
  
hidden_size = 128
n_layer = 8
min_value = 0
max_value = 20
array_size = 20
def save_shape():
   config =  GPTConfig(vocab_size=(max_value - min_value + 1), block_size=array_size * 2, n_layer=n_layer, n_head=8, n_embd=hidden_size
    ,standparam=True)
   delta_config = GPTConfig(vocab_size=(max_value - min_value + 1), block_size=array_size * 2, n_layer=n_layer, n_head=8, n_embd=hidden_size*2
    ,standparam=True)
   gpt = GPT(config)
   delta_gpt = GPT(delta_config)
   base_shapes = get_shapes(gpt)
   delta_shapes = get_shapes(delta_gpt)
   make_base_shapes(base_shapes, delta_shapes, savefile="./gpt_base_shape.bsh")
def get_infshapes(model,base):
    base_shapes = _extract_shapes(base)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    return infshapes
def fsdp_main(rank, world_size,width,args):
    setup(rank, world_size)
    epochs = 10
    #get datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if rank==0:
        import wandb
        
        run = wandb.init(
            # Set the project where this run will be logged
            project="gpt-test",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{width}_{current_datetime}",
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "gpt-moe",
            "dataset": "sorting_array",
            "epochs": epochs,
            "min_value":min_value,
            "max_value":max_value,
            "array_size":array_size,
            "hidden_size":width,
            "n_layer":n_layer
            })


    def filter_module_by_name(name:str):
        # if '0' in name and 'transformer' in name:
        #     return True
        return True

    # Create Dataset and DataLoader
    dataset = SortingDataset(num_samples=50000, array_size=array_size, min_value=min_value, max_value=max_value)
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=int(32/world_size), sampler=sampler)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    config =GPTConfig(vocab_size=(max_value - min_value + 1), block_size=array_size * 2, n_layer=n_layer, n_head=8, n_embd=width
        ,standparam=not args.mup)
    torch.cuda.set_device(rank)
    def _init_weights(module):
        if isinstance(module, torch.nn.Linear):
            if getattr(module, "init_by_n_layer", False):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # Initialize model
    with torch.device("meta"):
        model = GPT(config)

    fully_shard(model)
    model.to_empty(device=rank)
    model.apply(_init_weights)
    if args.mup:
        set_base_shapes(model, args.base_shape_path)
    if args.mup:
        infshapes = get_infshapes(model, args.base_shape_path)
        optimizer = Adam(model.named_parameters(),infshapes, lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    df = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        for batch_idx,batch in enumerate(data_loader,1):
            if rank == 0:
                remove_hooks = []
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(module.register_forward_hook(
                        _record_coords(df, width, name, batch_idx,
                            output_fdict=None,
                            input_fdict=None,
                            param_fdict=None)))
            inputs, labels = batch['input'].to(rank), batch['label'].to(rank)
            x = torch.concat([inputs, labels], dim=-1)
            optimizer.zero_grad()
            logits, loss = model(x[:, :-1], x[:, 1:].clone())
            loss.backward()
            optimizer.step()
            if rank == 0:
                for handle in remove_hooks:
                    handle.remove()
            if batch_idx == 500:
                if rank ==  0:
                    import json
                    json.dump(df,open(f'./stat_{rank}_{width}.json','w'))
                exit(0)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if rank==0:
            epoch_duration = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s')

            run.log({"avg_loss": avg_loss,"epoch_time":epoch_duration})

    def evaluate_model_fsdp(model, rank, world_size, num_tests=100, batch_size=32, verbose=False):
        model.eval()
        
        test_dataset = SortingDataset(num_samples=num_tests * batch_size, array_size=array_size, 
                                    min_value=min_value, max_value=max_value)
        test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

        correct = 0
        total = 0

        with torch.no_grad(),FSDP.summon_full_params(model):
            for batch in test_loader:
                inputs, labels = batch['input'].to(rank), batch['label'].to(rank)
                generated = model.generate(inputs, max_new_tokens=array_size, temperature=1.0, top_k=1)
                predictions = generated[:, array_size:]
                expected = labels

                batch_correct = torch.sum(torch.all(expected == predictions, dim=1)).item()
                correct += batch_correct
                total += inputs.size(0)

                if verbose or batch_correct == 0:
                    for i in range(inputs.size(0)):
                        print('Test Input:', inputs[i].cpu().numpy())
                        print('Expected  sorted:', expected[i].cpu().numpy())
                        print('Model Prediction:', predictions[i].cpu().numpy())
                        print('-----')

        accuracy_tensor = torch.tensor([correct, total], dtype=torch.float32, device=rank)
        dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_correct, total_samples = accuracy_tensor.tolist()
            accuracy = total_correct / total_samples
            print(f'Accuracy over {int(total_samples)} tests: {accuracy:.2%}')
            run.log({"accuracy": accuracy})
    num_evaluations = (100//world_size)*world_size
    evaluate_model_fsdp(model,rank,world_size,num_tests=num_evaluations, batch_size=1, verbose=False) 
    # Run evaluation
    if rank == 0:
        run.finish()
        cleanup()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    
    parser.add_argument("--make_base_shape",action="store_true")
    parser.add_argument("--mup",action="store_true")
    parser.add_argument("--base_shape_path",type=str,default=None)
    args = parser.parse_args()
    if args.make_base_shape:
        save_shape()
        exit(0)
    WORLD_SIZE = torch.cuda.device_count()
    import numpy as np
    widths = 2**np.arange(7, 12)
    for width in widths:
        mp.spawn(fsdp_main,
            args=(WORLD_SIZE,int(width),args),
            nprocs=WORLD_SIZE,
            join=True)