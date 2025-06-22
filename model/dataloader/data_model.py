import lightning as L
from pathlib import Path

from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader

# Annoying
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    from tokenizer.tokrnizer import Tokenizer
    from dataset_py import TongShunDataset
else:
    from tokenizer.tokrnizer import Tokenizer
    from .dataset_py import TongShunDataset

@dataclass
class DataConfig:
    train_datas: List[str] = field(default_factory=lambda: ["./train_datas"])
    val_datas: List[str] = field(default_factory=lambda: ["./val_datas"])

    voca_path: str = "./model/dict.txt"
    tokenizer: str = "./model/tokenizer/tokenizer.json"
    chinese_only: bool = True
    negative_sample_rate: int = 3
    batch_size: int = 16

class TongShunDataModule(L.LightningDataModule):
    def __init__(self, conf: DataConfig):
        '''
        datas: Both folders and files can be passed in
        '''
        super().__init__()
        self.train_datas = self._get_data_paths(conf.train_datas)
        self.val_datas = self._get_data_paths(conf.val_datas)
        self.conf = conf

        self.tokenizer = Tokenizer(conf.tokenizer)

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
            self.train_dataset = TongShunDataset(self.train_datas, self.voca, self.conf.chinese_only, self.conf.negative_sample_rate, self.tokenizer.encode)
            self.val_dataset = TongShunDataset(self.val_datas, self.voca, self.conf.chinese_only, self.conf.negative_sample_rate, self.tokenizer.encode, val_mode=True)
        elif stage == "validate":
            self.val_dataset = TongShunDataset(self.val_datas, self.voca, self.conf.chinese_only, self.conf.negative_sample_rate, self.tokenizer.encode, val_mode=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.conf.batch_size)
