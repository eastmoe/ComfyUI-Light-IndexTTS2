import torch
from transformers import Wav2Vec2BertModel

from indextts.utils.compat import torch_load
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec


def build_semantic_model(path_="./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt", model_name_or_path="facebook/w2v-bert-2.0"):
    semantic_model = Wav2Vec2BertModel.from_pretrained(model_name_or_path)
    semantic_model.eval()
    stat_mean_var = torch_load(path_, map_location="cpu")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg):
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec
