## my_pipeline

### 今までのコンペで使用してきたコードを集めたもの

### How to install
```bash
pip install git+https://github.com/ju-ki/my_pipeline
```


### Set up environment
```python
from jukijuki.utils.logger import Logger
from jukijuki.utils.util import create_folder, seed_everything

class Config:
    competition_name = "hogehoge"
    exp_name = "hoge"
    seed = 42

create_folder(Config)
seed_everything(Config.seed)
logger = Logger(Config.log_dir, Config.exp_name)
```
