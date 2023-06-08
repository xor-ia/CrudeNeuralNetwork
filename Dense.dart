import "Matrix.dart";
import "Layer.dart";

class Dense extends Layer {
	late Matrix weights;
	late Matrix biases;

	Dense(int inp, int out) {
		//                           M   N
		this.weights = Matrix.rand(inp, out);
		this.biases = Matrix.zeros(1, out);
	}

	@override
	Matrix forward(Matrix inp) {
		this.input = inp.copy(); // ? x M
		return inp.matMul(this.weights).addVec(this.biases);
	}

	@override
	Matrix backward(Matrix gradients, double lr) {
		Matrix dW = this.input.T().matMul(gradients); // M x ? * ? x N -> M x N == shape of weight
		Matrix dB = gradients.sumRows();

		this.weights = this.weights.sub(dW.mulScalar(lr));
		this.biases = this.biases.sub(dB.mulScalar(lr));

		return gradients.matMul(this.weights.T());
	}
}