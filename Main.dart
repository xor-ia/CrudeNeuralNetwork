import "Matrix.dart";
import "dart:math";


double sigmoid(double x) {
	return 1/(1+exp(-x));
}
double D_sigmoid(double x) {
	double sigX = sigmoid(x);
	return sigX * (1 - sigX);
}

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

	// initialize network parameters
	//   network size : 2 (input) -> 3 (hidden_0) -> 1 (output)
	Matrix w0 = Matrix.rand(2, 3);
	Matrix b0 = Matrix.zeros(1, 3);

	Matrix w1 = Matrix.rand(3, 1);
	Matrix b1 = Matrix.zeros(1, 1);


	for (int _ = 0; _ < 10000; _++) {
		// forward pass
		Matrix z0 = x.matMul(w0).addVec(b0);  // n x 2 * 2 x 3 -> n x 3
		Matrix a0 = z0.map(sigmoid);
		
		Matrix z1 = a0.matMul(w1).addVec(b1); // n x 3 * 3 x 1 -> n x 1
		Matrix a1 = z1.map(sigmoid);

		print("MSE : " + MSE(y, a1).toString());

		// backward pass
		Matrix lossGrad = D_MSE(y, a1); // n x 1
		Matrix a1Grad = z1.map(D_sigmoid).mulElement(lossGrad);

		Matrix w1Grad = a0.T().matMul(a1Grad); // 3 x n * n x 1 -> 3 x 1
		Matrix b1Grad = a1Grad.sumRows(); // 1 x 1

		Matrix z1Grad = a1Grad.matMul(w1.T()); // n x 1 * 1 x 3 -> n x 3
		Matrix a0Grad = z0.map(D_sigmoid).mulElement(z1Grad); // n x 3

		Matrix w0Grad = x.T().matMul(a0Grad); // 2 x n * n x 3 -> 2 x 3
		Matrix b0Grad = a0Grad.sumRows(); // 1 x 3

		// update parameters
		w1 = w1.sub(w1Grad.mulScalar(lr));
		b1 = b1.sub(b1Grad.mulScalar(lr));

		w0 = w0.sub(w0Grad.mulScalar(lr));
		b0 = b0.sub(b0Grad.mulScalar(lr));
	}
}

