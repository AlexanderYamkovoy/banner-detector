from flask import request, send_from_directory
from flask_restx import marshal, Resource
from werkzeug.utils import secure_filename

from app import app, api
from src import parsers, serializers
from ad_insertion_executor import ad_insertion_executor


@api.route('/conf')
class ConfigurationResource(Resource):
    @staticmethod
    def get() -> dict:
        """ Get current model configuration """

        return marshal(app.config['model_config'], serializers.conf_serializer)

    @api.expect(serializers.conf_serializer)
    def put(self) -> dict:
        """ Update model configuration """

        app.config['model_config'].update(request.get_json())
        app.save_conf()
        return marshal(app.config['model_config'], serializers.conf_serializer)


@api.route('/processing')
class ProcessingResource(Resource):
    @staticmethod
    def get() -> list:
        """ Get files list """

        return [file.name for file in app.files_path.glob('**/*')]

    @api.expect(parsers.files_parser)
    def post(self):
        """ Process new files, return processed video """

        logo, video = request.files['logo'], request.files['video']

        logo_filename = secure_filename(logo.filename)
        video_filename = secure_filename(video.filename)

        logo_path = str(app.files_path / logo_filename)
        video_path = str(app.files_path / video_filename)

        logo.save(logo_path)
        video.save(video_path)

        video_name = ad_insertion_executor(video_path, logo_path, str(app.conf_path))

        return send_from_directory(app.root_path,
                                   video_name,
                                   as_attachment=True,
                                   attachment_filename=video_name)


@api.route('/processing/<path:filename>')
class ProcessingInstanceResource(Resource):
    @staticmethod
    def get(filename: str):
        """ Get file by name """

        return send_from_directory(app.files_path,
                                   filename,
                                   as_attachment=True,
                                   attachment_filename=filename)
