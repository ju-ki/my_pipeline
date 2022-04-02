## my_pipeline

### 今までのコンペで使用してきたコードを集めたもの

### How to install
```bash
pip install git+https://github.com/ju-ki/my_pipeline
```


### Set up environment for tabular competition
```python
from jukijuki.utils.logger import Logger
from jukijuki.utils.timer import Timer
from jukijuki.utils.util import create_folder, seed_everything
from jukijuki.validation.SturgesRuleStratifiedKFold import sturges_skf
from jukijuki.tabular.util import AbstractBaseBlock, WrapperBlock, run_blocks
from jukijuki.tabular.feature_engine import LabelEncodingBlock, CountEncodingBlock, AggregationBlock, OneHotEncodingBlock, CrossCategoricalFeatureBlock
from jukijuki.gb_model.model_lgbm import MyLGBMModel
from jukijuki.gb_model.model_xgboost import MyXGBModel
from jukijuki.gb_model.model_cat import MyCatModel

class Config:
    competition_name = "hogehoge"
    exp_name = "hoge"
    target_col = "target"
    seed = 42
    n_fold = 5

create_folder(Config)
seed_everything(Config.seed)
logger = Logger(Config.log_dir, Config.exp_name)
```

### Set up environment for image competition
```python
from jukijuki.image.util import get_file_path
from jukijuki.utils.timer import Timer
from jukijuki.utils.logger import Logger
from jukijuki.utils.EarlyStopping import EarlyStopping
from jukijuki.utils.util import create_folder, seed_everything, get_device
from jukijuki.validation.SturgesRuleStratifiedKFold import sturges_skf
from jukijuki.pytorch_model.util import get_optimizer, get_scheduler

class Config:
    apex=False
    competition_name = "hogehoge"
    exp_name = "hoge"
    target_col = "target"
    batch_size = 32
    num_workers = 4
    size = 224
    epochs = 8
    model_name = "resnet34d"
    optimizer_name = "AdamW"
    scheduler = "CosineAnnealingLR"
    T_max = epochs
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    n_fold = 5
    seed = 42
    target_size = 1

create_folder(Config)
seed_everything(Config.seed)
device = get_device()
logger = Logger(Config.log_dir, Config.exp_name)
```


### Set up environment for nlp competition
```python
from jukijuki.nlp.util import get_tokenizer
from jukijuki.utils.timer import Timer
from jukijuki.utils.logger import Logger
from jukijuki.utils.EarlyStopping import EarlyStopping
from jukijuki.utils.util import create_folder, seed_everything, get_device
from jukijuki.validation.SturgesRuleStratifiedKFold import sturges_skf
from jukijuki.pytorch_model.util import get_optimizer, get_scheduler

class Config:
    apex=False
    competition_name = "hogehoge"
    exp_name = "hoge"
    target_col = "target"
    sentence_col = "hoge"
    batch_size = 32
    num_workers = 4
    max_len = 250
    epochs = 8
    model_name = "roberta-base"
    optimizer_name = "AdamW"
    scheduler = "CosineAnnealingLR"
    T_max = epochs
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    n_fold = 5
    seed = 42
    target_size = 1

create_folder(Config)
seed_everything(Config.seed)
device = get_device()
tokenizer = get_tokenizer(Config.model_name)
logger = Logger(Config.log_dir, Config.exp_name)
```