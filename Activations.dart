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



class Sigmoid extends Layer {
	@override
	Matrix forward(Matrix inp) {
		this.input = inp.copy();
		return inp.map(sigmoid);
	}

	@override
	Matrix backward(Matrix gradients, double lr) {
		return this.input.map(D_sigmoid).mulElement(gradients);
	}
}


class ReLU extends Layer {
	@override
	Matrix forward(Matrix inp) {
		this.input = inp.copy();
		return inp.map(relu);
	}

	@override
	Matrix backward(Matrix gradients, double lr) {
		return this.input.map(D_relu).mulElement(gradients);
	}
}


