# import json
# from api import app
# from ariadne import load_schema_from_path, make_executable_schema, \
#     graphql_sync, snake_case_fallback_resolvers, ObjectType
# from ariadne.constants import PLAYGROUND_HTML
# from flask import request, jsonify
# from api.queries import ???
#
# query = ObjectType("Query")
# query.set_field("??", ???)
#
# type_defs = load_schema_from_path("schema.graphql")
# schema = make_executable_schema(
#     type_defs, query, snake_case_fallback_resolvers
# )
#
# @app.route("/graphql", methods=["GET"])
# def graphql_playground():
#     return PLAYGROUND_HTML, 200
#
# @app.route("/graphql", methods=["POST"])
# def graphql_server():
#     data = request.get_json()
#     success, result = graphql_sync(
#         schema,
#         data,
#         context_value=request,
#         debug=app.debug
#     )
#     status_code = 200 if success else 400
#     print(jsonify(result))
#     return jsonify(result), status_code
#
#
# # delete these lines after studying and implementing to avoid code crashing
# # replace ?? query name in schema.graphql file and repalce ??? with the function name imported from the api.queries in line 7