import "dart:math";

class Matrix {
	late List<List<double>> mat;
	late int rows, cols;

	Matrix(List<List<double>> mat, int rows, int cols) {
		this.mat = mat;
		this.rows = rows;
		this.cols = cols;
	}

	Iterable<(int, int)> iterate() sync* {
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.cols; j++) {
				yield (i, j);
			}
		}
	}
	static Matrix load(List<List<double>> mat) {
		// assume that the input is always valid. cuz im lazy
		return Matrix(mat, mat.length, mat[0].length);
	}
	static Matrix zeros(int rows, int cols) {
		return Matrix(List.generate(rows, (_) => List.generate(cols, (_) => 0)), rows, cols);
	}
	static Matrix rand(int rows, int cols) {
		Random rng = Random();
		List<List<double>> results = [];
		for (int i = 0; i < rows; i++) {
			List<double> rows = [];
			for (int j = 0; j < cols; j++) {
				rows.add(rng.nextDouble());
			}
			results.add(rows);
		}
		return Matrix(results, rows, cols);
	}

	Matrix T() {
		Matrix toRet = Matrix.zeros(this.cols, this.rows);
		for (var (i, j) in this.iterate()) {
			toRet.mat[j][i] = this.mat[i][j];
		}
		return toRet;
	}
	
	Matrix copy() {
		Matrix toRet = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			toRet.mat[i][j] = this.mat[i][j];
		}
		return toRet;
	}
	Matrix matMul(Matrix b) {
		if (this.cols != b.rows) {
			throw ("Cannot multiply matrix A:" + this.strShape() + " and B:"+ b.strShape());
		}
		Matrix output = Matrix.zeros(this.rows, b.cols);
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < b.cols; j++) {
				double sum = 0;
				for (int k = 0; k < this.cols; k++) {
					sum += this.mat[i][k] * b.mat[k][j];
				}
				output.mat[i][j] = sum;
			}
		}
		return output;
	} 
	Matrix mulScalar(double x) {
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = this.mat[i][j] * x;
		}
		return results;
	}
	Matrix mulElement(Matrix b) {
		if (this.strShape() != b.strShape()) {
			throw ("Cannot multiply element-wise matrix A:" + this.strShape() + " and B:" + b.strShape());
		}
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = this.mat[i][j] * b.mat[i][j];
		}
		return results;
	}


	String strShape() {
		return this.rows.toString() + "x" + this.cols.toString();
	}
	void dumps() {
		print("Shape : " + this.rows.toString() + "x" + this.cols.toString());
		String toPrint = "";
		for (int i = 0; i < this.rows; i++) {
			toPrint = toPrint + "| ";
			for (int j = 0; j < this.cols; j++) {
				toPrint = toPrint + this.mat[i][j].toString() + ", ";
			}
			toPrint = toPrint.substring(0, toPrint.length - 1);
			toPrint = toPrint + "|\n";
		}
		print(toPrint.substring(0, toPrint.length - 1));
	}
	Matrix add(Matrix b) {
		if (this.strShape() != b.strShape()) {
			throw ("Cannot add matrix A:" + this.strShape() + " and B:" + b.strShape());
		}
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = this.mat[i][j] + b.mat[i][j];
		}
		return results;
	}
	Matrix addVec(Matrix b) {
		if (b.rows != 1) {
			throw ("B:" + b.strShape() + " is not a vector!");
		} else if (this.cols != b.cols) {
			throw ("Cannot add matrix to vector A:" + this.strShape() + " and B:" + b.strShape());
		}
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = this.mat[i][j] + b.mat[0][j];
		}
		return results;
	}
	Matrix sub(Matrix b) {
		if (this.strShape() != b.strShape()) {
			throw ("Cannot subtract matrix A:" + this.strShape() + " and B:" + b.strShape());
		}
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = this.mat[i][j] - b.mat[i][j];
		}
		return results;
	}
	Matrix map(double Function(double) f) {
		Matrix results = Matrix.zeros(this.rows, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[i][j] = f(this.mat[i][j]);
		}
		return results;
	}
	
	double sum() {
		double sum = 0;
		for (var (i, j) in this.iterate()) {
			sum += this.mat[i][j];
		}
		return sum;
	}
	Matrix sumRows() {
		Matrix results = Matrix.zeros(1, this.cols);
		for (var (i, j) in this.iterate()) {
			results.mat[0][j] += this.mat[i][j];
		}
		return results;
	}

}