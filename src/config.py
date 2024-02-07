from omegaconf import OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    train_path: str
    test_path: str
    data_csv_path: str
    submission_path: str
    device: str
    model_arch: str
    result_model_path: str
    scaler_path: str

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
