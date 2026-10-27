from comfy_api.latest import ComfyExtension

from .src.comfy_light_indextts2.checkpoint_nodes import CHECKPOINT_NODES


class IndexTTSExtension(ComfyExtension):
    async def get_node_list(self):
        return list(CHECKPOINT_NODES)


async def comfy_entrypoint():
    return IndexTTSExtension()
