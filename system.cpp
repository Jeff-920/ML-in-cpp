// ����ѧϰ�����Ĵ���ʵ�֣��ִ꣩
// �񾭲��࣬���ݴ��������ࣩ����ʧ�����࣬�Ż������࣬ͼ�ο��ӻ������࣬ѵ��������
// �񾭲��ࣺ���Բ�OK��relu��OK������OK
// ���ݴ�������ı��Σ�����ʱ���Ƿ���㿼�����������������ȡ����д������OK�����������OK��������ӷֽ��֪ʶ��
// ��ʧ��������������ʧOK�����������ʧOK��ƽ�����������ʧOK��Huber��ʧ������ʱ�俼���Ƿ���ӣ�
// �Ż�����������ݶ��½�->�����㣬Adam���о�����
// ͼ�λ����Ӵ��ڣ���ʾ����ͼ��������ͼ���ܷ�ʵʱ����
// ѵ�����ԣ�ѵ����train��Ҫ��װ�����ԡ�>k�۲���

// ����->��ʧ����OK->�Ż�����->�񾭲�->
#include <initializer_list> 
#include <graphics.h>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <random>
#include <cmath>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;


// ������
class Matrix {
	// 
protected:
	vector<vector<double>> data;
	int rows;
	int cols;
public:
	// ��ʼ������
	Matrix(const initializer_list<initializer_list<double>> &list) {
		rows = list.size();
		cols = (rows > 0) ? list.begin()->size() : 0; // ���������������� 0����ôͨ�� list.begin() ��ȡ����ʼ���б�ĵ�һ��Ԫ��
		data.reserve(rows); // Ԥ���ռ䣬�������·���
		for (const auto& row_list : list) {
			data.emplace_back(row_list);  // ��ÿ�����б�ת��Ϊ vector<int>
			// emplace_back ����ֱ��ʹ�ô���Ĳ����������ڲ��������
		}
	}
	Matrix(int r, int c) :rows(r), cols(c) {
		data.resize(rows, vector<double>(cols,0));
	}
	Matrix(){}

	// ��λ����
	
	// ��ӡ����
	void showMatrix()const {
		for (const auto& row : data) {
			for (double element : row) {
				cout << element<<" ";
			}
			cout << endl;
		}
	}

	void showShape()const {
		cout << "(" << rows << "," << cols << ")" << endl;
	}

	// ��ȡ���������ĺ���
	int getRows() const {
		return rows;
	}

	// ��ȡ���������ĺ���
	int getCols() const {
		return cols;
	}

	int getElement(int i, int j)const {
		if (i >= 0 && i < rows && j >= 0 && j < cols) {
			return data[i][j];
		}
		return 0;
	}

	// ����[]
	// ����ÿһ��λ�õ�Ԫ��ֵ
	// ����ֵ�����ã�����ֱ���޸�
	vector<double>& operator[](int index) {
		return data[index];
	}

	// ���ؼӷ�
	Matrix operator + (double other) {
		Matrix sum(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				sum.data[i][j] = this->data[i][j] + other;
			}
		}
		return sum;
	}

	Matrix operator + (Matrix& other) {
		if (this->rows == other.rows && this->cols == other.cols) {
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] + other.data[i][j];
				}
			}
			return sum;
		}

		// �㲥����
		// �ӷ��Ĺ㲥���ƣ� �൱��ȱʧ��ά�ȶ�������b
		// �й㲥
		else if (this->rows == other.rows && other.cols == 1){
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] + other.data[i][0];
				}
			}
			return sum;
		}
		// �й㲥
		else if (other.rows == 1 && this->cols == other.cols) {
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] + other.data[0][j];
				}
			}
			return sum;
		}
	}

	// ���ؼ���
	Matrix operator - (Matrix& other) {
		if (this->rows == other.rows && this->cols == other.cols) {
			Matrix sub(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sub.data[i][j] = this->data[i][j] - other.data[i][j];
				}
			}
			return sub;
		}
		else if (this->rows == other.rows && other.cols == 1) {
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] - other.data[i][0];
				}
			}
			return sum;
		}
		else if (other.rows == 1 && this->cols == other.cols) {
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] - other.data[0][j];
				}
			}
			return sum;
		}
	}

	// ���س˷�
	Matrix operator * ( Matrix& other)const {
		if (this->cols == other.rows) {
			Matrix mul(this->rows, other.cols);
			for (int i = 0; i < this->rows; i++) {
				for (int j = 0; j < other.cols; j++) {
					for (int z = 0; z < other.rows; z++) {
						mul[i][j] += this->data[i][z] * other[z][j];
					}
				}
			}
			return mul;
		}
		else {
			printf("False");
			throw invalid_argument("False");
		}
	}

	//������������
	Matrix operator * (double scalar)const {
		Matrix result(this->rows, this->cols);
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				result[i][j] = scalar * this->data[i][j];
			}
		}
		return result;
	}	

	// ת��
	Matrix transpose(Matrix a) {
		Matrix result(a.getCols(), a.getRows());
		for (int i = 0; i < a.getCols(); i++) {
			for (int j = 0; j < a.getRows(); j++) {
				result[i][j] = a[j][i];
			}
		}
		return result;
	}
};

// ��ʧ����
class Loss {
public:
	virtual double calculate(Matrix tru, Matrix pre) = 0;
	virtual ~Loss() {};
};

// ��������ʧ
class CrossEntropyLoss : public Loss {
public:
	double calculate(Matrix pre, Matrix tru) {
		double loss = 0;
		for (int i = 0; i < tru.getRows(); i++) {
			for (int j = 0; j < tru.getCols(); j++) {
				loss += -1 * (tru[i][j] * log(pre[i][j])) - (1 - tru[i][j]) * log(1 - pre[i][j]);
			}
		}
		return loss / tru.getCols() * tru.getRows();
	}
};

// MSE
class MSE :public Loss {
public:
	double calculate(Matrix pre, Matrix tru) {
		double loss = 0;
		for (int i = 0; i < tru.getRows(); i++) {
			for (int j = 0; j < tru.getCols(); j++) {
				loss += (tru[i][j] - pre[i][j]) * (tru[i][j] - pre[i][j]);
			}
		}
		return loss / tru.getCols() * tru.getRows();
	}
};

// MAE
class MAE : public Loss {
public:
	double calculate(Matrix pre, Matrix tru) {
		double loss = 0;
		for (int i = 0; i < tru.getRows(); i++) {
			for (int j = 0; j < tru.getCols(); j++) {
				loss += abs(tru[i][j] - pre[i][j]);
			}
		}
		return loss / tru.getCols() * tru.getRows();
	}
};

// �񾭲���
class NN {
public:
	virtual Matrix function(Matrix input) = 0;
	virtual ~NN() {};
	virtual Matrix& get_w() = 0;
	virtual Matrix& get_b() = 0;
};

// ���Բ�
class Linear :public NN {
private:
	Matrix w_in;
	Matrix b_in;
public:
	// ���캯�������г�ʼ��
	Linear(int input_dim, int output_dim) {
		
		// ��ʼ��w b
		w_in = Matrix(input_dim, output_dim);
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> dis(-1.0, 1.0);
		for (int i = 0; i < input_dim; i++) {
			for (int j = 0; j < output_dim; j++) {
				w_in[i][j] = dis(gen);
			}
		}
		b_in = Matrix(1, output_dim);
		for (int j = 0; j < output_dim; j++) {
			b_in[0][j] = dis(gen);
		}	
	}

	// ǰ�򴫲�
	Matrix function(Matrix input) {
		Matrix output = input * w_in + b_in;
		return output;
	}
	// Ҫ��ȫʵ��NN��ĳ�����
	Matrix& get_w() {
		return w_in;
	}

	Matrix& get_b() {
		return b_in;
	}
};
// override һ���ؼ���������ʾ�ı���һ���麯��������������д�˻����麯��

// sigmod�� 
class Sigmod : public NN {
private:
	Linear layer;
public:
	Sigmod(int input_dim, int output_dim) :layer(input_dim, output_dim) {};
	Matrix function(Matrix input) {
		Matrix linear_output = layer.function(input);
		
		Matrix output(linear_output.getRows(), linear_output.getCols());
		for (int j = 0; j < output.getRows(); j++) {
			for (int i = 0; i < output.getCols(); i++) {
				output[j][i] = 1 / (1 + exp(-linear_output[j][i]));
			}
		}
		return output;
	}

	Matrix& get_w() {
		return layer.get_w();
	}

	Matrix& get_b() {
		return layer.get_b();
	}
};

// relu��
class Relu :public NN {
private:
	Linear layer;
public:
	Relu(int input_dim, int output_dim) :layer(input_dim, output_dim) {};
	Matrix function(Matrix input) {
		Matrix linear_output = layer.function(input);

		Matrix output(linear_output.getRows(), linear_output.getCols());
		for (int j = 0; j < output.getRows(); j++) {
			for (int i = 0; i < output.getCols(); i++) {
				output[j][i] = output[j][i] > 0 ? output[j][i] : 0;
			}
		}
		return output;
	}

	Matrix& get_w() {
		return layer.get_w();
	}

	Matrix& get_b() {
		return layer.get_b();
	}
};

//������������������ǰ��ͷ��򴫲�
class Sequential {
	// �����񾭲�ָ��
private:
	vector<NN*> layers;
	int layer_num = 0;
public:
	void add_layer(NN* layer) {
		layers.push_back(layer);
		layer_num += 1;
	}

	Matrix forward(Matrix input) {
		Matrix output = input;
		for (auto& layer : layers) {
			output = layer->function(output);
		}
		return output;
	}

	Matrix backward(Matrix input, Matrix target, Matrix output,double lr) {
		Matrix error = output - target;
		for (int i = layer_num - 1; i >= 0; i--) {
			auto* layer = layers[i];
			Matrix& w = layer->get_w();
			Matrix& b = layer->get_b();
			// �����ݶ�
			Matrix dj_db(1, error.getCols());
			for (int j = 0; j < error.getCols(); j++) {
				for (int k = 0; k < error.getRows(); k++) {
					dj_db[0][j] += error[k][j];
				}
			}
			Matrix grad_w = input.transpose(input) * error;
			printf("OK\n");
			// ����Ȩ�غ�ƫ��
			for (int r = 0; r < w.getRows(); ++r) {
				for (int c = 0; c < w.getCols(); ++c) {
					w[r][c] -= lr * grad_w[r][c];
				}
			}
			//printf("OK1\n");
			for (int c = 0; c < b.getCols(); ++c) {
				b[0][c] -= lr * dj_db[0][c];
			}
			printf("OK\n");
			// ����error����
			if (i > 0) {
				Matrix w_t = w.transpose(w);
				if (typeid(*layer) == typeid(Sigmod)) {
					// ��
					for (int r = 0; r < error.getRows(); r++) {
						for (int c = 0; c < error.getCols(); c++) {
							double value = 1 / (1 + exp(-input[r][c]));
							error[r][c] *= value * (1 - value);
						}
					}
				}
				else if (typeid(*layer) == typeid(Relu)) {
					for (int r = 0; r < error.getRows(); r++) {
						for (int c = 0; c < error.getCols(); c++) {
							error[r][c] *= (input[r][c] > 0 ? 1 : 0);
						}
					}
				}
			}
			// ����input
			// ���õ�ǰ���
			printf("OK\n");
			input = layer->function(input);
			printf("OK\n");
		}
		
		return error;
	}
};


int main(){
	Matrix input{ {0.1, 0.3, 0.7}, {0.8, 0.9, 0.4}, {0.8, 0.6, 0.7} };
	Matrix target{ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };

	// ����������
	Sequential model;
	model.add_layer(new Linear(3, 4));
	model.add_layer(new Sigmod(4, 4));
	model.add_layer(new Linear(4, 3));
	model.add_layer(new Sigmod(3, 3));

	// ǰ�򴫲�
	Matrix output = model.forward(input);

	// ������ʧ
	CrossEntropyLoss loss;
	double loss_value = loss.calculate(output, target);
	cout << "Loss: " << loss_value << endl;

	// ���򴫲�
	model.backward(input, target, output, 0.01);

	return 0;
}
