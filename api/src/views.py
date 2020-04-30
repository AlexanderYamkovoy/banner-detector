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

        logo_path = app.files_path / logo_filename
        video_path = app.files_path / video_filename
        report_path = app.files_path / f'{video_filename}_report.txt'

        logo.save(logo_path)
        video.save(video_path)

        with open(str(report_path), 'w') as report_file:
            # TODO: rewrite video file instead of writing the result to 'result.avi'

            ad_insertion_executor(str(video_path), str(logo_path), str(app.conf_path), report_file)

        return send_from_directory(app.files_path, video_filename)


@api.route('/processing/<path:filename>')
class ProcessingInstanceResource(Resource):
    @staticmethod
    def get(filename: str) -> list:
        """ Get file by name """

        return send_from_directory(app.files_path, filename)
