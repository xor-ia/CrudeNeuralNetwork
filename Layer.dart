import "Matrix.dart";

class Layer {
	late Matrix input;
	late Matrix output;
	Matrix forward(Matrix inp) {throw UnimplementedError;}
	Matrix backward(Matrix gradients, double lr) {throw UnimplementedError;}
}