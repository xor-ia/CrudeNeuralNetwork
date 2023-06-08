import "Matrix.dart";
import "Layer.dart";
import "dart:math";

double sigmoid(double x) {
	return 1/(1+exp(-x));
}
double D_sigmoid(double x) {
	double sigX = sigmoid(x);
	return sigX * (1 - sigX);
}

double relu(double x) {
	return x > 0 ? x : 0;
}
double D_relu(double x) {
	return x > 0 ? 1 : 0;
}

class Activation extends Layer {
	late double Function(double) ac_forward;
	late double Function(double) ac_backward;
	@override
	Matrix forward(Matrix inp) {
		this.input = inp.copy();
		return inp.map(this.ac_forward);
	}

	@override
	Matrix backward(Matrix gradients, double lr) {
		return this.input.map(this.ac_backward).mulElement(gradients);
	}
}

class Sigmoid extends Activation {
	Sigmoid() {
		this.ac_forward = sigmoid;
		this.ac_backward = D_sigmoid;
	}
}


class ReLU extends Activation {
	ReLU() {
		this.ac_forward = relu;
		this.ac_backward = D_relu;
	}
}

