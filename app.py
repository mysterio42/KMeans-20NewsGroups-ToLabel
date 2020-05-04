from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from utils.model import LabelMe

app = Flask(__name__)
api = Api(app)


class Clusterize(Resource):

    def post(self):
        posted_data = request.get_json()

        assert 'sentences' in posted_data
        assert 'n_clusters' in posted_data

        labelme = LabelMe(posted_data['sentences'], posted_data['n_clusters'])
        labelme.train(labelme.embed())
        labeled_data = labelme.clusterize()
        del labelme
        return jsonify(labeled_data)


api.add_resource(Clusterize, '/clusterize')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
