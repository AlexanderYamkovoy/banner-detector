from flask_restx import reqparse
from werkzeug.datastructures import FileStorage

files_parser = reqparse.RequestParser()

files_parser.add_argument(
    'logo',
    type=FileStorage,
    location='files',
    help='Logo for processing',
    required=True
)

files_parser.add_argument(
    'video',
    type=FileStorage,
    location='files',
    help='Video file for processing',
    required=True
)
