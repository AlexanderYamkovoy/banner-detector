from flask_restx import fields
from app import api

conf_serializer = api.model(
    'Configuration',
    {
        'kernel': fields.Integer(readonly=True),
        'min_area_threshold': fields.Integer,
        'max_area_threshold': fields.Integer(readonly=True),
        'perimeter_threshold': fields.Float(readonly=True),
        'corners_count': fields.Integer(readonly=True),
        'field_threshold': fields.Integer(readonly=True),
        'contour_threshold': fields.Float,
        'dst_threshold': fields.Integer(readonly=True),
        'window': fields.Integer(readonly=True),
        'poly_order': fields.Integer(readonly=True),
    }
)
