import yaml


class AttrDict(dict):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        mapping = {} if mapping is None else mapping
        self.update(mapping, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def update(self, mapping=None, **kwargs):
        items = {}
        if mapping is not None:
            items.update(mapping)
        items.update(kwargs)
        for key, value in items.items():
            self[key] = value

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return AttrDict(yaml.safe_load(file) or {})
