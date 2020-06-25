import os
import torch
import logging
import logging.handlers


class LogManager(object):
    def __init__(self, checkpoint_step, save_dir, model_name, log_file_name, mode='min', device=None):
        super(LogManager, self).__init__()
        self.checkpoint_step = checkpoint_step
        self.save_dir = save_dir
        self.mode = mode
        self.device = device
        self.best_value = float("inf") if mode == "min" else float("-inf")

        os.makedirs(save_dir, exist_ok=True)
        self.ckpt_path_name = os.path.join(self.save_dir, model_name)
        self.log_file_name = os.path.join(self.save_dir, log_file_name)

        # logging.basicConfig(level=logging.INFO,
        #                     format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        self.logger = logging.getLogger(model_name)
        console_handle = logging.StreamHandler()
        console_handle.setLevel(logging.INFO)
        file_handle = logging.FileHandler(self.log_file_name, 'w', encoding='utf-8')
        file_handle.setLevel(logging.INFO)
        self.logger.addHandler(console_handle)
        # self.logger.addHandler(file_handle)

    def log(self, step, epoch, model, value=None, info='', is_train=True):
        if is_train:
            self._save_state(epoch, model, value)
            self.logger.info('| train_step: %4d | loss: %.6f | %s' % (step, value, info))
        else:
            self.logger.info('| eval:       %4d | loss: %.6f | %s' % (step, value, info))

    def restore_state_at_step(self, model, step=0):
        if os.path.isfile(self.path_name + "_%d.ckpt", step):
            state_dict = torch.load(self.path_name + "_%d.ckpt" % step,
                                    map_location=self.device)
            model.load_state_dict(state_dict, strict=True)
        else:
            print('Checkpoint not found! Model remain unchanged!')
        return model

    def restore_best_state(self, model):
        if os.path.isfile(self.ckpt_path_name + "_best.ckpt"):
            state_dict = torch.load(self.ckpt_path_name + "_best.ckpt",
                                    map_location=self.device)
            model.load_state_dict(state_dict, strict=True)
        else:
            print('Best Checkpoint not found! Model remain unchanged!')
        return model

    def _save_state(self, epoch, model, value=None):
        if epoch % self.checkpoint_step == 0:
            torch.save(model.state_dict, self.ckpt_path_name + "_%d.ckpt" % epoch)
        if self.mode == "min" and value < self.best_value:
            torch.save(model.state_dict, self.ckpt_path_name + "_best.ckpt")
        elif self.mode == "max" and value > self.best_value:
            torch.save(model.state_dict, self.ckpt_path_name + "_best.ckpt")
