"""
Learning schedulers to be used
Author: Hangyu Li
Date: 09/05/2022
"""
import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LambdaLR


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class OurReduceLROnPlateau(ReduceLROnPlateau):
    """
    Our implementation only add one method get_last_lr() to fit in transformer trainer
    """
    def __init__(self, *args, **kwargs):
        super(OurReduceLROnPlateau, self).__init__(*args, **kwargs)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        # Check whether in linear warmup
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def update_lr(self):
        """
        Only update the last lr to fit in transformer's trainer
        """
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        """
        Return also the last computed learning rate
        """
        return self.get_last_lr()


class OurExponentialLR(ExponentialLR):
    """
    Our implementation add min_lr to enable the scheduler not to decay to zero
    """
    def __init__(self, optimizer, gamma, min_lr=5e-5, last_epoch=-1, verbose=False):
        super(OurExponentialLR, self).__init__(optimizer, gamma, last_epoch, verbose)
        self.min_lr = min_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]

        ret_lr = []
        for group in self.optimizer.param_groups:
            if group['lr'] * self.gamma >= self.min_lr:
                ret_lr.append(group['lr'] * self.gamma)
            else:
                ret_lr.append(self.min_lr)

        return ret_lr


    def _get_closed_form_lr(self):
        ret_lr = []
        for base_lr in self.base_lrs:
            if base_lr * self.gamma ** self.last_epoch >= self.min_lr:
                ret_lr.append(base_lr * self.gamma ** self.last_epoch)
            else:
                ret_lr.append(self.min_lr)

        return ret_lr


def get_multistep_lr_scheduler(optimizer, last_epoch=-1):
    lr_init = optimizer.defaults["lr"]
    def lr_lambda(current_epoch):
        # As LambdaLR itself gets multiplied by lr_init, we divide it here
        if current_epoch < 20:
            return lr_init / lr_init
        elif 20 <= current_epoch < 50:
            return lr_init / 10 / lr_init
        elif 50 <= current_epoch < 100:
            return lr_init / 20 / lr_init
        else:
            return lr_init / 100 / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)