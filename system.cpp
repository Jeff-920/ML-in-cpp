// 机器学习初步的代码实现（手搓）
// 神经层类，数据处理（矩阵类），损失函数类，优化函数类，图形可视化窗口类，训练测试类
// 神经层类：线性层OK，relu层OK，Σ层OK
// 数据处理：矩阵的变形（根据时间是否充足考虑张量），矩阵的提取（重写索引）OK，矩阵的运算OK（考虑添加分解的知识）
// 损失函数：交叉熵损失OK，均方误差损失OK，平均绝对误差损失OK，Huber损失（根据时间考虑是否添加）
// 优化函数：随机梯度下降->求导运算，Adam（有精力）
// 图形化可视窗口：显示折线图，最后的热图（能否实时？）
// 训练测试：训练，train主要封装，测试—>k折测试

// 矩阵->损失函数OK->优化函数->神经层->
#include <initializer_list> 
#include <graphics.h>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <random>
#include <cmath>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;


// 矩阵处理
class Matrix {
	// 
protected:
	vector<vector<double>> data;
	int rows;
	int cols;
public:
	// 初始化矩阵
	Matrix(const initializer_list<initializer_list<double>> &list) {
		rows = list.size();
		cols = (rows > 0) ? list.begin()->size() : 0; // 如果矩阵的行数大于 0，那么通过 list.begin() 获取外层初始化列表的第一个元素
		data.reserve(rows); // 预留空间，避免重新分配
		for (const auto& row_list : list) {
			data.emplace_back(row_list);  // 将每个子列表转换为 vector<int>
			// emplace_back 可以直接使用传入的参数在容器内部构造对象
		}
	}
	Matrix(int r, int c) :rows(r), cols(c) {
		data.resize(rows, vector<double>(cols,0));
	}
	Matrix(){}

	// 单位矩阵
	
	// 打印矩阵
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

	// 获取矩阵行数的函数
	int getRows() const {
		return rows;
	}

	// 获取矩阵列数的函数
	int getCols() const {
		return cols;
	}

	int getElement(int i, int j)const {
		if (i >= 0 && i < rows && j >= 0 && j < cols) {
			return data[i][j];
		}
		return 0;
	}

	// 重载[]
	// 设置每一个位置的元素值
	// 返回值是引用，可以直接修改
	vector<double>& operator[](int index) {
		return data[index];
	}

	// 重载加法
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

		// 广播机制
		// 加法的广播机制， 相当于缺失的维度都加上了b
		// 列广播
		else if (this->rows == other.rows && other.cols == 1){
			Matrix sum(rows, cols);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					sum.data[i][j] = this->data[i][j] + other.data[i][0];
				}
			}
			return sum;
		}
		// 行广播
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

	// 重载减法
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

	// 重载乘法
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

	//重载数乘运算
	Matrix operator * (double scalar)const {
		Matrix result(this->rows, this->cols);
		for (int i = 0; i < this->rows; i++) {
			for (int j = 0; j < this->cols; j++) {
				result[i][j] = scalar * this->data[i][j];
			}
		}
		return result;
	}	

	// 转置
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

// 损失函数
class Loss {
public:
	virtual double calculate(Matrix tru, Matrix pre) = 0;
	virtual ~Loss() {};
};

// 交叉熵损失
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

// 神经层类
class NN {
public:
	virtual Matrix function(Matrix input) = 0;
	virtual ~NN() {};
	virtual Matrix& get_w() = 0;
	virtual Matrix& get_b() = 0;
};

// 线性层
class Linear :public NN {
private:
	Matrix w_in;
	Matrix b_in;
public:
	// 构造函数，进行初始化
	Linear(int input_dim, int output_dim) {
		
		// 初始化w b
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

	// 前向传播
	Matrix function(Matrix input) {
		Matrix output = input * w_in + b_in;
		return output;
	}
	// 要完全实现NN里的抽象类
	Matrix& get_w() {
		return w_in;
	}

	Matrix& get_b() {
		return b_in;
	}
};
// override 一个关键字用于显示的表明一个虚函数在派生类中重写了基类虚函数

// sigmod层 
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

// relu层
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

//神经网络容器，并进行前向和反向传播
class Sequential {
	// 定义神经层指针
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
			// 计算梯度
			Matrix dj_db(1, error.getCols());
			for (int j = 0; j < error.getCols(); j++) {
				for (int k = 0; k < error.getRows(); k++) {
					dj_db[0][j] += error[k][j];
				}
			}
			Matrix grad_w = input.transpose(input) * error;
			printf("OK\n");
			// 更新权重和偏差
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
			// 更新error函数
			if (i > 0) {
				Matrix w_t = w.transpose(w);
				if (typeid(*layer) == typeid(Sigmod)) {
					// 求导
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
			// 更新input
			// 调用当前层的
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

	// 创建神经网络
	Sequential model;
	model.add_layer(new Linear(3, 4));
	model.add_layer(new Sigmod(4, 4));
	model.add_layer(new Linear(4, 3));
	model.add_layer(new Sigmod(3, 3));

	// 前向传播
	Matrix output = model.forward(input);

	// 计算损失
	CrossEntropyLoss loss;
	double loss_value = loss.calculate(output, target);
	cout << "Loss: " << loss_value << endl;

	// 反向传播
	model.backward(input, target, output, 0.01);

	return 0;
}
