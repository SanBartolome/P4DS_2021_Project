from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model.joblib')
labels = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
	return """
	<html>
		<head>
			<meta charset="UTF-8" />
			<meta name="viewport" content="width=device-width, initial-scale=1" />
			<meta http-equiv="X-UA-Compatible" content="ie-edge" />
			<title>Web Grupo 4 - Formulario</title>
			<link
			href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
			rel="stylesheet"
			/>
		</head>
		<body class="h-100 bg-dark">
			<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
			<div class="h-100 w-100 d-flex justify-content-center align-items-center">
				<div class="card w-50">
					<div class="card-body py-4">
						<h2 class="card-title text-center">IRIS predicción</h2>
						<h4 class="card-title text-center">Ingrese los siguientes valores:</h4>
						<form class="d-flex flex-column" action='predict' method='GET'>
							<div class="mb-3">
								<label for="sepalLength" class="form-label">Longitud del cáliz</label>
								<input type="number" id="sepalLength" class="form-control w-50" required min=0>
							</div>
							<div class="mb-3">
								<label for="sepalWidth" class="form-label">Ancho del cáliz</label>
								<input type="number" id="sepalWidth" class="form-control w-50" required min=0>
							</div>
							<div class="mb-3">
								<label for="petalLength" class="form-label">Largo del pétalo</label>
								<input type="number" id="petalLength" class="form-control w-50" required min=0>
							</div>
							<div class="mb-3">
								<label for="petalWidth" class="form-label">Ancho del pétalo</label>
								<input type="number" id="petalWidth" class="form-control w-50" required min=0>
							</div>
							<button type="submit" class="btn btn-primary mt-4 w-50 align-self-center">Predecir</button>
						</form>
					</div>
				</div>
			</div>
		</body>
	</html>
	"""

@app.route("/predict")
def predict():
	v1 = float(request.args.get('sepalLength'))
	v2 = float(request.args.get('sepalWidth'))
	v3 = float(request.args.get('petalLength'))
	v4 = float(request.args.get('petalWidth'))

	result = model.predict(np.array([[v1,v2,v3,v4]]))

	return "<h1> Prediction: {}</h1>".format(labels[result[0]])

if __name__ == '__main__':
	app.run(debug=False, use_reloader=True)