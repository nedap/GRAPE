from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .logger import print_log
from termcolor import colored

def test_dataloader(module: LightningDataModule, terminal_logger, to_test):
    max_length = max(
        len('is DataLoader instance'),
        len('contains data'),
        len('batch size < len(data)')
    )

    def format_message(description: str, status: str) -> str:
        padded_description = description.ljust(max_length)
        return f"[TEST] {padded_description}: {status}"

    def is_dl_instance(dl: DataLoader) -> bool:
        passed = isinstance(dl, DataLoader)
        status = colored('Passed', 'green') if passed else colored('Failed', 'red')
        message = format_message('is DataLoader instance', status)
        print_log(message, logger=terminal_logger)
        return passed

    def contains_data(dl: DataLoader) -> bool:
        passed = len(dl.dataset) > 0
        status = colored('Passed', 'green') if passed else colored('Failed', 'red')
        message = format_message('contains data', status)
        print_log(message, logger=terminal_logger)
        return passed

    def check_bs(dl: DataLoader) -> bool:
        passed = len(dl.dataset) > dl.batch_size
        status = colored('Passed', 'green') if passed else colored('Failed', 'red')
        message = format_message('batch size < len(data)', status)
        print_log(message, logger=terminal_logger)
        return passed

    module.setup('fit')
    for test in to_test:
        print_log(f'Testing the {test} DataLoader')
        if test == 'train':
            dl = module.train_dataloader()
        elif test == 'val':
            dl = module.val_dataloader()
        else:
            raise NotImplemented(f'Test for {test} not implemented.')
    
        tests = [is_dl_instance, contains_data, check_bs]
        for test in tests:
            if not test(dl):
                raise Exception(f'{colored("DataLoader", "blue")} test {colored("Failed", "red")}.')
