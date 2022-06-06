from loguru import logger


class Config:
    def __init__(self):
        self._configs = {}

    def register_from_file(self, name, params):
        if self._configs.get(name) is not None:
            return params

        self._configs[name] = params
        logger.info(f"{name} config has registered...")

        return params

    def get(self, name):
        try:
            return self._configs[name]

        except KeyError:
            raise KeyError(f"{name} does not exist")


config = Config()
