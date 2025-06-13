import lightning as L
from pathlib import Path

from dataset_py import TongShunDataset
from dataclasses import dataclass
from torch.utils.data import DataLoader

# Annoying
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    from tokenizer.tokrnizer import Tokenizer
else:
    from ..tokenizer.tokrnizer import Tokenizer

@dataclass
class config:
    train_datas: list = ["./train_datas"]
    val_datas: list = ["./val_datas"]

    voca_path: str = "./model/dict.txt"
    chinese_only: bool = True
    negative_sample_rate: int = 3
    batch_size: int = 16

class TongShunDataModule(L.LightningDataModule):
    def __init__(self, conf: config):
        '''
        datas: Both folders and files can be passed in
        '''
        super().__init__()
        self.train_datas = self._get_data_paths(conf.train_datas)
        self.val_datas = self._get_data_paths(conf.val_datas)
        self.conf = conf

        self.tokenizer = Tokenizer(conf.voca_path)

        with open(conf.voca_path, 'r', encoding='utf-8') as file:
            self.voca = file.read().splitlines()

    def _get_data_paths(self, datas:list[str]) -> list[Path]:
        paths = []
        for i in datas:
            x = Path(i)
            if x.is_file() and x.suffix == ".txt":
                paths.append(x)
            elif x.is_dir():
                paths.extend(x.glob("**/*.txt"))
        return paths

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = TongShunDataset(self.train_datas, self.voca, self.conf.chinese_only, self.conf.negative_sample_rate)
        elif stage == "validate":
            self.val_dataset = TongShunDataset(self.val_datas, self.voca, self.conf.chinese_only, self.conf.negative_sample_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.conf.batch_size)
