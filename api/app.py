import yaml

from flask import Flask
from flask_restx import Api

from pathlib import Path

from src import settings


class AdvApp(Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_path = Path.cwd()
        self.files_path = self.root_path / settings.files_path

        if not self.files_path.exists():
            self.files_path.mkdir()

        self.conf_path = self.root_path / settings.conf_path
        self.config['model_config'] = self.load_conf()

    def load_conf(self) -> dict:
        with open(self.conf_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def save_conf(self) -> None:
        with open(self.conf_path, 'w') as file:
            return yaml.dump(self.config['model_config'], file)


app = AdvApp(__name__)
api = Api(app, title='Advertisement REST API', validate=True)

from src.views import *

if __name__ == '__main__':
    app.run(debug=True)
