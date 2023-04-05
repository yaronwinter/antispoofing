import torch
import numpy as np
import time
import copy
import aasist
import data_utils
import utils as gen_utils
from torch.utils.data import DataLoader
import tester

DEFAULT_MAX_EER = 1000
SEED = 42

class Trainer:
    def __init__(self):
        self.active_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def train(self, config: dict) -> tuple:
        print('AASIST trainer - start')
        log_file = open(config["log_file_name"], "w", encoding="utf-8")

        pending_model = aasist.Model(config)
        pending_model = pending_model.to(self.active_device)
        optimal_model = None

        print("Load samples")
        train_dataset = data_utils.Dataset_ASVspoof2019(config, "train")
        dev_dataset = data_utils.Dataset_ASVspoof2019(config, "dev")
        eval_dataset = data_utils.Dataset_ASVspoof2019(config, "eval")

        gen = torch.Generator()
        gen.manual_seed(SEED)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=gen_utils.seed_worker,
                                  generator=gen)
        
        dev_loader = DataLoader(dev_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)

        
        print('set optimizer & loss')
        optim_config = config["optim_config"]
        optim_config["epochs"] = config["num_epochs"]
        optim_config["steps_per_epoch"] = len(train_loader)
        optimizer, scheduler = gen_utils.create_optimizer(pending_model.parameters(), optim_config)

        weight = torch.FloatTensor([0.1, 0.9]).to(self.active_device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        best_dev_eer = DEFAULT_MAX_EER
        best_dev_epoch = -1
        best_eval_eer = DEFAULT_MAX_EER
        best_eval_epoch = -1
        
        print('start training loops. #epochs = ' + str(config['num_epochs']))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train EER':^11} | {'Dev EER':^10} | {'Eval EER':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train EER':^11} | {'Dev EER':^10} | {'Eval EER':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
        log_file.flush()
            
        
        min_loss = 100
        num_no_imp = 0
        for i in range(config['num_epochs']):
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            
            pending_model.train()
            for signals, labels in train_loader:
                signals = signals.to(self.active_device)
                labels = labels.to(self.active_device)
                
                _, logits = pending_model(signals)
                
                optimizer.zero_grad()
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation test.
            dev_eer, _ = tester.compute_eer(pending_model, dev_loader)
            train_eer, _ = tester.compute_eer(pending_model, train_loader)
            eval_eer, _ = tester.compute_eer(pending_model, eval_loader)
            print(f"{epoch:^7} | {avg_loss:^12.6f} | {train_eer:^9.2f} | {dev_eer:^9.2f} |  {eval_eer:^9.4f} | {epoch_time:^9.2f}")
            log_file.write(f"{epoch:^7} | {avg_loss:^12.6f} | {train_eer:^9.2f} | {dev_eer:^9.2f} |  {eval_eer:^9.4f} | {epoch_time:^9.2f}\n")
            log_file.flush()
                
            if avg_loss < min_loss:
                min_loss = avg_loss
                num_no_imp = 0
            else:
                num_no_imp += 1
                
            if num_no_imp > config["early_stop_max_no_imp"]:
                print('early stop exit')
                log_file.write('\tEarly Stop exit\n')
                log_file.flush()
                break
            
            if epoch < config["min_valid_epochs"]:
                continue
            
            if dev_eer < best_dev_eer:
                best_dev_eer = dev_eer
                best_dev_epoch = epoch
                optimal_model = copy.deepcopy(pending_model)

            if eval_eer < best_eval_eer:
                best_eval_eer = eval_eer
                best_eval_epoch = epoch
        
        print('AASIST trainer - end\n')
        print("Best Dev EER = {:.2f}".format(best_dev_eer) + ", best epoch = " + str(best_dev_epoch))
        print("Best Eval Acc = {:.2f}".format(best_eval_eer) + ", best epoch = " + str(best_eval_epoch))
        log_file.write("Best Dev EER = {:.2f}".format(best_dev_eer) + ", best epoch = " + str(best_dev_epoch) + "\n")
        log_file.write("Best Eval Acc = {:.2f}".format(best_eval_eer) + ", best epoch = " + str(best_eval_epoch) + "\n")
        log_file.close()
        return pending_model, optimal_model, best_dev_epoch
