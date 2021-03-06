// Neural network . Simple library to create neural networks and make them learn using labeled data.

#include<iostream>
#include<stdlib.h>
#include<cmath>
#include<conio.h>
#include<fstream>
#include<iomanip>
#include<vector>

using namespace std;
#define DATA_SIZE 3 // IGNORE this
#define Features 3  // IGNORE this
#define DATAFILE "maDATA.dat"
#define DATAFILE_net "netDATA.dat"
#define ERROR -1 

int SIZE = DATA_SIZE; // IGNORE this

double approx(double a)
{
	if (abs(a) < 0.000001)
		a = 0;

	return a;
}

class matrix
{
private:
	// create a dynamically allocated array of pointers each of which pointing to another dynamically allocated array = 2D ARRAY 
	double trace;
	double det;
	int id;

	void createid()
	{
		fstream file;
		file.open(DATAFILE, ios::in | ios::binary);

		if (file.is_open())
		{
			file.seekg(-1 * (int)sizeof(matrix), ios::end);
			file.read((char *)this, sizeof(matrix));
			id++;
		}
		else
		{
			id = 1;
		}
		file.close();
	}

protected:

	int rows;
	int columns;
	double* *m;

	double cofactor(int i, int j)
	{
		if (rows != columns)
		{
			return -1;
		}

		if (rows == 1)
		{
			return m[0][0];
		}

		double ** a = new double *[rows - 1];

		for (int i = 0; i < columns - 1; i++)
		{
			a[i] = new double[columns - 1];
		}

		int temp1, temp2;

		for (int l = 0; l < rows; l++)
			for (int n = 0; n < columns; n++)
			{
				temp1 = l;
				temp2 = n;
				if (l != i && n != j)
				{
					if (l > i)
					{
						temp1--;
					}
					if (n > j)
					{
						temp2--;
					}

					a[temp1][temp2] = m[l][n];
				}
			}

		matrix t(a, rows - 1, columns - 1);

		return pow(-1, (i + j)) * t.getdet();
	}

public:

	matrix(double **a, int row, int col)
	{
		rows = row;
		columns = col;
		m = new double*[rows];

		for (int i = 0; i < rows; i++)
		{
			m[i] = new double[columns];
		}

		m = a;
		id = 0;
		// createid(); // matrixes can have the same id if some are not saved. so better to not call this. id = 0 means no id
	}

	matrix(int row = 1, int col = 1)
	{
		rows = row;
		columns = col;

		m = new double*[rows];

		for (int i = 0; i < rows; i++)
		{
			m[i] = new double[columns];
		}

		id = 0;
	}

	int load(int i) // i is the id
	{
		fstream file;
		file.open(DATAFILE, ios::in | ios::binary);
		if (!file.is_open())
		{
			file.close();
			return ERROR;
		}
		file.seekg((i - 1) * sizeof(matrix), ios::beg);
		file.read((char *)this, sizeof(matrix));
		file.close();
		return 0;
	}

	int getid()
	{
		return id;
	}

	int save()
	{
		createid();
		fstream infile;

		infile.open(DATAFILE, ios::app | ios::binary);

		clean();

		infile.write((char *)this, sizeof(matrix));
		infile.close();
	}

	void clean()
	{
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				m[i][j] = approx(m[i][j]);
			}
	}

	int ftrace()
	{
		if (rows != columns)
		{
			trace = ERROR;
			return ERROR;
		}

		trace = 0;

		for (int i = 0; i < rows; i++)
		{
			trace += m[i][i];
		}
		return 0;
	}

	double gettrace()
	{
		return trace;
	}

	int getrows()
	{
		return rows;
	}

	int getcols()
	{
		return columns;
	}

	double getdet()
	{
		return det;
	}

	double element(int i, int j)
	{
		return m[i][j];
	}

	void find_det()
	{
		if (rows != columns)
		{
			det = ERROR;
			return;
		}

		det = 0;

		if (rows == 1)
		{
			det = m[0][0];
			return;
		}

		for (int i = 0; i < rows; i++)
		{
			det += m[0][i] * cofactor(0, i);
		}

		clean();
	}

	void cofactormatrix()
	{
		if (rows != columns)
		{
			return;
		}

		double ** cm = new double*[rows];

		for (int i = 0; i < columns; i++)
		{
			cm[i] = new double[columns];
		}

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < columns; j++)
			{
				cm[i][j] = cofactor(i, j);
			}
		}

		m = cm;
	}

	matrix T()
	{
		double **r = new double*[columns];
		for (int i = 0; i < columns; i++)
		{
			r[i] = new double[rows];
		}
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				r[j][i] = m[i][j];
			}
		matrix temp(r, columns, rows);

		return temp;
	}

	int inverse()
	{
		find_det();
		if (det == 0 || rows != columns)
		{
			cout << " NOT INVERTIBLE! " << endl;
			return ERROR;
		}

		matrix temp = *this;
		temp.cofactormatrix();
		temp = temp.T();

		temp = temp.scalar(1.0 / det);

		*this = temp;
	}

	static double** convert(double a[DATA_SIZE][Features], int r, int s)
	{
		double **x = new double*[r];

		for (int i = 0; i < r; i++)
		{
			x[i] = new double[s];
		}

		for (int j = 0; j < r; j++)
			for (int i = 0; i < s; i++)
			{
				x[j][i] = a[j][i];
			}

		return x;
	}

	void print()
	{
		clean();
		for (int i = 0; i < rows; i++)
		{
			cout << endl;
			for (int j = 0; j < columns; j++)
				cout << m[i][j] << " ";
		}
	}

	void modify(int i, int j, double value)
	{
		m[i][j] = value;
	}

	matrix scalar(double value)
	{
		matrix c = *this;

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
			{
				c.modify(i, j, value * m[i][j]);
			}

		return c;
	}

	static matrix multi(matrix a, matrix b, int &flag )
	{
		double t = 0;

		matrix c = matrix(a.getrows(), b.getcols());

		if (a.getcols() != b.getrows())
		{
			flag = ERROR;
			return c;
		}

		for (int i = 0; i < a.getrows(); i++)
		{
			for (int j = 0; j < b.getcols(); j++)
			{
				for (int k = 0; k < a.getcols(); k++)
				{
					t += a.element(i, k) * b.element(k, j);
				}
				c.modify(i, j, t);
				t = 0;
			}
		}
		flag = 0;
		return c;
	}

	static matrix multi_element(matrix a, matrix b, int &flag )
	{

		matrix c = a;

		if (a.getrows() != b.getrows() || a.getcols() != b.getcols())
		{
			flag = ERROR;
			return c;
		}

		for (int i = 0; i < a.getrows(); i++)
		{
			for (int j = 0; j < a.getcols(); j++)
			{
				c.modify(i, j, a.element(i, j) * b.element(i, j));
			}
		}

		flag = 0;
		return c;
	}

	

	

	static matrix add(matrix a, matrix b)
	{
		if (a.getrows() == b.getrows() && a.getcols() == b.getcols())
		{
			matrix c = a; // just random initializasion
			for (int i = 0; i < a.getrows(); i++)
				for (int j = 0; j < a.getcols(); j++)
					c.modify(i, j, a.element(i, j) + b.element(i, j));
			return c;
		}
		else
			return matrix::null();
	}

	static matrix null()
	{
		matrix temp(1, 1);
		return temp;
	}
};

class vec : public matrix
{
	int size;
	double mag;

public:

	vec(double *v, int s) : matrix(s, 1)
	{
		rows = s;
		columns = 1;

		size = rows;
		m = new double*[rows];

		for (int i = 0; i < rows; i++)
		{
			m[i] = new double[columns];
		}

		for (int i = 0; i < size; i++)
		{
			m[i][0] = v[i];
		}

	}

	vec(int s = 1) : matrix(s, 1)
	{
		rows = s;
		columns = 1;

		size = rows;
		m = new double*[rows];

		for (int i = 0; i < rows; i++)
		{
			m[i] = new double[columns];
		}
	}

	int getsize()
	{
		return size;
	}

	void find_magnitude()
	{
		mag = 0;
		for (int i = 0; i < size; i++)
		{
			mag += pow(m[i][0], 2);
		}
		mag = pow(mag, 0.5);
	}

	double getmag()
	{
		return mag;
	}

	static bool compare(vec a, vec b)
	{
		bool result = true;
		if (a.getsize() == b.getsize())
		{
			for (int i = 0; i < a.getsize(); i++)
			{
				if (a.element(i, 0) != b.element(i, 0))
					result = false;
			}
		}
		else
		{
			result = false;
		}

		return result;
	}

	static vec add(vec a, vec b)
	{
		vec c = a;
		if (a.getsize() == b.getsize())
		{
			for (int i = 0; i < c.getsize(); i++)
				c.modify(i, 0, a.element(i, 0) + b.element(i, 0));
		}

		return c;
	}

	vec scalar(double value)
	{
		vec c = *this;
		for (int i = 0; i < size; i++)
			c.modify(i, 0, m[i][0] * value);

		return c;
	}

	static vec multi_element(vec a, vec b, int &flag)
	{

		vec c = a;

		if (a.getrows() != b.getrows())
		{
			flag = ERROR;
			return c;
		}

		for (int i = 0; i < a.getrows(); i++)
		{
			c.modify(i, 0, a.element(i, 0) * b.element(i, 0));
		}

		flag = 0;
		return c;
	}

	static vec multi(matrix a, vec b, int &flag)
	{
		double t = 0;

		vec c = vec(a.getrows());
		if (a.getcols() != b.getrows())
		{
			flag = ERROR;
			return c;
		}

		for (int i = 0; i < a.getrows(); i++)
		{
			for (int j = 0; j < b.getcols(); j++)
			{
				for (int k = 0; k < a.getcols(); k++)
				{
					t += a.element(i, k) * b.element(k, j);
				}
				c.modify(i, j, t);
				t = 0;
			}
		}
		flag = 0;
		return c;
	}
};

class neuralnetwork
{
private:

	int l;
	vector<matrix> theta;
	vector<matrix> bigdel;
	vector<matrix> J_p;
	vector<vec> net;

	vector<vec> del;

	matrix X;
	matrix result;

	double lambda;
	double alpha;
	int max_iter;

public:

	neuralnetwork(vector<matrix> init_theta) 
	{
		theta = init_theta;

		l = init_theta.size() + 1;

			bigdel = theta;

		for (int i = 0; i < l - 1; i++)
		{
			for (int m = 0; m < bigdel[i].getrows(); m++)
				for (int n = 0; n < bigdel[i].getcols(); n++)
					bigdel[i].modify(m, n, 0);
		}
	}

	void forwardpropagation(vec x, vector<int> B)
	{
		// x is given without a bias , bias vector B is provided independantly with
		// length L-1
		net.resize(l);
		net[0] = x;

		for (int i = 0; i < l - 1; i++)
		{
			vec temp(net[i].getsize() + 1);

			for (int j = 1; j < temp.getsize(); j++)
			{
				temp.modify(j, 0, net[j].element(j, 0));
			}
			temp.modify(0, 0, B[i]);
			net[i] = temp;

			int b = 0;
			net[i + 1] = sigmoid(vec::multi(theta[i], net[i] , b));
		}
	}

	void error(vec y)
	{
		// net is along with the biases
		del.resize(l);
		y = y.scalar(-1);
		del[l] = vec::add(net[l], y);

		for (int p = l - 1; p > 0; p++)
		{
			if (p + 1 != l)
			{
				del[p + 1] = remove_bias(del[p + 1]);
			}

			vec temp = net[p];
			for (int i = 0; i < temp.getsize(); i++)
				temp.modify(i, 0, 1 - temp.element(i, 0));
			int b = 0;
			del[p] = vec::multi_element(vec::multi_element(vec::multi(theta[p].T(), del[p + 1] , b), net[p] , b), temp , b);
		}

	}

	void J_prime(int  m)
	{
		vector<int> B;

		for (int i = 0; i < l; i++)
			B.push_back(1);

		vec x = vec(X.getrows());
		vec y = vec(result.getrows());

		for (int p = 0; p < m; p++)
		{
			for (int i = 0; i < x.getsize(); i++)
			{
				x.modify(i, 0, X.element(i, p));
				y.modify(i, 0, result.element(i, p));
			}

			forwardpropagation(x, B);
			error(y);
			for (int j = 0; j < l - 1; j++)
			{
				matrix temp_del(del[j + 1].getrows(), 1); // del[j+1]
				matrix temp_net(net[j].getrows(), 1); // net[j]

				for (int i = 0; i < temp_del.getrows(); i++)
					temp_del.modify(i, 0, del[j + 1].element(i, 0));

				for (int i = 0; i < temp_net.getrows(); i++)
					temp_net.modify(i, 0, net[j].element(i, 0));
				int b = 0;
				bigdel[j] = matrix::add(bigdel[j], matrix::multi(temp_del, temp_net.T() , b));
			}

		}

		for (int j = 0; j < l - 1; j++)
		{
			J_p[j] = matrix::add(bigdel[j], theta[j].scalar(lambda)).scalar(1.0 / m);
		}

	}

	vec sigmoid(vec z)
	{
		int p = z.getsize();
		for (int i = 0; i < p; i++)
		{
			z.modify(i, 0, 1.0 / (1 + exp(-1 * z.element(i, 0) / 15.0)));
		}

		return z;
	}

	vec remove_bias(vec v)
	{
		vec b(v.getsize() - 1);

		for (int i = 1; i < v.getsize(); i++)
		{
			b.modify(i - 1, 0, v.element(i, 0));
		}

		return b;
	}

	void learn(int m, double lamb, double alp, int max_it)
	{
		lambda = lamb;
		alpha = alp;
		max_iter = max_it;

		// cost_saver = ones(max_iter , 1);

		for (int g = 0; g < max_iter; g++)
		{
			J_prime(m);
			for (int j = 0; j < l - 1; j++)
			{
				theta[j] = matrix::add(theta[j], J_p[j].scalar(-1 * alpha));
			}
			//cost_saver(g) = cost(X , theta , result , m);
		}

	}

	vector<int> save()
	{
		vector<int> model_id;
		for (int i = 0; i < l - 1; i++)
		{
			theta[i].save();
			model_id.push_back(theta[i].getid());
		}
		return model_id;
	}

	void load(vector<int> model_id)
	{
		for (int i = 0; i < l - 1; i++)
		{
			theta[i].load(model_id[i]);
		}
	}
	/* Matlab code for reference
	function J = cost( X , theta , result , m)
	J = 0;
	for i = 1:m
		net = forwardpropagation(X(: , i) , theta ,result(: , i));
		J = J + sum((net{size(theta , 1) + 1} - result(: , i)).^2);
	end
	J = J/m;
	end
	*/
};

struct netsaver
{
	vector<int> model_id;
	int id;

	void save()
	{
		fstream file;
		file.open(DATAFILE_net, ios::app | ios::binary);
		file.write((char*)this, sizeof(netsaver));
		file.close();
	}

	bool load()
	{
		int original_id = id;
		bool flag = false;
		fstream file;
		file.open(DATAFILE_net, ios::in | ios::binary);

		if (file.is_open())
		{
			while (!file.eof())
			{
				file.read((char*)this, sizeof(netsaver));

				if (id == original_id)
				{
					flag = true;
					break;
				}
			}
		}

		id = original_id;
		file.close();
		return flag;
	}

};

int main()
{
	/*
	neuralnetwork net1 = neuralnetwork(); // input parameters

	// do something with this network

	// saving the network 
	netsaver save1;
	save1.model_id = net1.save();
	save1.id = 23; // you can input any id for saving this network here
	save1.save();

	// loading from file
	netsaver save2;
	save2.id = 23; // enter your id here
		if (save2.load())
		{
			net1.load(save2.model_id);
		}
		*/
	return 0;
}