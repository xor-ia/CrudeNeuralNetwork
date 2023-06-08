import "Matrix.dart";
import "Dense.dart";
import "Layer.dart";
import "Activations.dart";


double MSE(Matrix yTrue, Matrix yPred) {
	return (yTrue.sub(yPred)).map((x) => x * x).sum() / (yPred.rows * yTrue.cols);
}
Matrix D_MSE(Matrix yTrue, Matrix yPred) {
	return (yPred.sub(yTrue)).mulScalar(2 / (yPred.rows * yTrue.cols));
}

void main() {
	Matrix x = Matrix.load([
		[1, 0],
		[0, 0],
		[1, 1],
		[0, 1],
	]);
	Matrix y = Matrix.load([
		[1],
		[0],
		[0],
		[1]
	]);
	double lr = 0.1;
	List<Layer> network = [
		Dense(2, 3),
		ReLU(),
		Dense(3, 1),
		Sigmoid()
	];


	for (int _ = 0; _ < 10000; _++) {

		Matrix outp = x.copy();
		// forward pass
		for (Layer layer in network) {
			outp = layer.forward(outp);
		}
		print("MSE : " + MSE(y, outp).toString());
		// backward pass
		Matrix gradient = D_MSE(y, outp);
		for (int i = network.length - 1; i >= 0; i--) {
			gradient = network[i].backward(gradient, lr);
		}
		
	}
}

