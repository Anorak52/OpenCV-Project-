#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>
#include <conio.h>
using namespace cv;
using namespace std;

typedef std::vector<std::vector<int>> Matrix;

int HowMuchThreshold(double** data, int rows, int cols, double porog)
{
	int count = 0;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (abs(data[i][j]) < porog)
				count++;
	return count;
}

void DeleteLessThreshold(double** data, int rows, int cols, double porog)
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (abs(data[i][j]) < porog)
				data[i][j] = 0;
}

int Difference(Mat image1, Mat image2)
{
	int count = 0;
	for (int i = 0; i < image1.rows; i++)
		for (int j = 0; j < image1.cols; j++)
			if (image1.at<uchar>(i, j) != image2.at<uchar>(i, j))
				count++;
	return count;
}



void PrintData(double** data, int rows, int cols)
{
	cout << endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << setw(5) << data[i][j] << " ";
		}
		cout << endl;
	}
}
void WaveletCols(double** data, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j += 2)
		{
			double tmp1, tmp2;
			tmp1 = (data[i][j] + data[i][j + 1]) / 2;
			tmp2 = (data[i][j] - data[i][j + 1]) / 2;
			data[i][j] = tmp1;
			data[i][j + 1] = tmp2;
		}
	}
}
void BackWaveletCols(double** data, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j += 2)
		{
			double tmp1, tmp2;
			tmp1 = data[i][j] + data[i][j + 1];
			tmp2 = data[i][j] - data[i][j + 1];
			data[i][j] = tmp1;
			data[i][j + 1] = tmp2;
		}
	}
}
void WaveletRows(double** data, int rows, int cols)
{
	for (int j = 0; j < cols; j++)
	{
		for (int i = 0; i < rows; i += 2)
		{
			double tmp1, tmp2;
			tmp1 = (data[i][j] + data[i + 1][j]) / 2;
			tmp2 = (data[i][j] - data[i + 1][j]) / 2;
			data[i][j] = tmp1;
			data[i + 1][j] = tmp2;
		}
	}
}
void BackWaveletRows(double** data, int rows, int cols)
{
	for (int j = 0; j < cols; j++)
	{
		for (int i = 0; i < rows; i += 2)
		{
			double tmp1, tmp2;
			tmp1 = data[i][j] + data[i + 1][j];
			tmp2 = data[i][j] - data[i + 1][j];
			data[i][j] = tmp1;
			data[i + 1][j] = tmp2;
		}
	}
}
void SeparateCols(double** data, int rows, int cols, int coef)
{
	double** tmp_data = new double* [rows];
	for (int i = 0; i < rows; i++)
		tmp_data[i] = new double[cols];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			tmp_data[i][j] = data[i][j];
	int move;
	for (int i = 0; i < rows / coef; i++) {
		move = 0;
		for (int j = 0; j < cols / coef; j += 2) {
			data[i][move] = tmp_data[i][j];
			data[i][cols / 2 / coef + move] = tmp_data[i][j + 1];
			move++;
		}
	}
	for (int i = 0; i < rows; i++)
		delete[] tmp_data[i];
	delete[] tmp_data;
}
void BackSeparateCols(double** data, int rows, int cols, int coef)
{
	double** tmp_data = new double* [rows];
	for (int i = 0; i < rows; i++)
		tmp_data[i] = new double[cols];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			tmp_data[i][j] = data[i][j];
	int move;
	for (int i = 0; i < rows / coef; i++) {
		move = 0;
		for (int j = 0; j < cols / coef; j += 2) {
			data[i][j] = tmp_data[i][move];
			data[i][j + 1] = tmp_data[i][cols / 2 / coef + move];
			move++;
		}
	}
	for (int i = 0; i < rows; i++)
		delete[] tmp_data[i];
	delete[] tmp_data;
}
void SeparateRows(double** data, int rows, int cols, int coef)
{
	double** tmp_data = new double* [rows];
	for (int i = 0; i < rows; i++)
		tmp_data[i] = new double[cols];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			tmp_data[i][j] = data[i][j];
	int move;
	for (int j = 0; j < cols / coef; j++) {
		move = 0;
		for (int i = 0; i < rows / coef; i += 2) {
			data[move][j] = tmp_data[i][j];
			data[rows / 2 / coef + move][j] = tmp_data[i + 1][j];
			move++;
		}
	}
	for (int i = 0; i < rows; i++)
		delete[] tmp_data[i];
	delete[] tmp_data;
}

void BackSeparateRows(double** data, int rows, int cols, int coef)
{
	double** tmp_data = new double* [rows];
	for (int i = 0; i < rows; i++)
		tmp_data[i] = new double[cols];
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			tmp_data[i][j] = data[i][j];
	int move;
	for (int j = 0; j < cols / coef; j++) {
		move = 0;
		for (int i = 0; i < rows / coef; i += 2) {
			data[i][j] = tmp_data[move][j];
			data[i + 1][j] = tmp_data[rows / 2 / coef + move][j];
			move++;
		}
	}
	for (int i = 0; i < rows; i++)
		delete[] tmp_data[i];
	delete[] tmp_data;
}

Mat DataToMat(double** data, int rows, int cols)
{
	Mat image(rows, cols, CV_8UC1);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			image.at<uchar>(i, j) = (uchar)abs(data[i][j]);
	return image;
}

//=============================================================

void Merge(cv::Mat& image, int reg1, int reg2, int nI, int end_x, int end_y, Matrix& A, int count) {
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j) {
			if (A[i][j] == reg1 || A[i][j] == reg2) {
				image.at<cv::Vec3b>(i, j)[0] = nI;
				image.at<cv::Vec3b>(i, j)[1] = nI;
				image.at<cv::Vec3b>(i, j)[2] = nI;
				A[i][j] = count;
			}
			if (i == end_x && j == end_y) return;
		}
}

int ToGray(const cv::Vec3b& color) {
	return (color[0] + color[1] + color[2]) / 3;

}


int fib(int n) {
	if (n <= 1)
		return n;
	return fib(n - 1) + fib(n - 2);
}

void histPull(const cv::Mat& sourse, int* hist, unsigned char& count) {
	for (int i = 0; i < 256; ++i)
		hist[i] = 0;
	for (int i = 0; i < sourse.rows; ++i)
		for (int j = 0; j < sourse.cols; ++j)
			hist[sourse.at<cv::Vec3b>(i, j)[0]]++;
	for (int i = 0; i < 256; ++i)
		if (hist[i] != 0) count++;
}

void strconcat(char* str, char* paterncode, char add) {
	int i = 0;
	while (*(paterncode + i) != '\0') {
		*(str + i) = *(paterncode + i);
		++i;
	}
	str[i] = add;
	str[i + 1] = '\0';
}

void probabilitisOfIntensity(int* hist, double* res, int countOfPixels) {
	for (int i = 0; i < 256; i++)
		res[i] = static_cast<double>(hist[i]) / static_cast<double>(countOfPixels);
}

void Monochrome(const cv::Mat& input, cv::Mat& output) {
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j) {
			cv::Vec3b colorIn = input.at<cv::Vec3b>(i, j);
			unsigned short colorOut =
				colorIn[0] * 0.0721 + colorIn[1] * 0.7154 + colorIn[2] * 0.2125;
			output.at<cv::Vec3b>(i, j) = cv::Vec3b(colorOut, colorOut, colorOut);
		}
}

void HuffmanCompression(const cv::Mat& input, cv::Mat output) {

	input.copyTo(output);
	Monochrome(input, output);
	int hist[256];
	unsigned char nodes = 0;
	histPull(output, hist, nodes);

	double probilitis[256];
	probabilitisOfIntensity(hist, probilitis, output.rows * output.cols);
	unsigned char maxLen = 0;
	double tmp = probilitis[0];
	for (int i = 1; i < 256; ++i)
		if (probilitis[i] < tmp && probilitis[i] > 0) tmp = probilitis[i];
	while (1 / tmp > fib(maxLen)) ++maxLen;
	maxLen -= 3;

	struct PixelInfo {
		unsigned char intens;
		double prob;
		PixelInfo* left, * right;
		char* code;
		void setSize(unsigned char len) { code = new char[len]; }
	};

	struct HuffCode {
		unsigned char intens;
		int arrloc;
		double prob;
	};

	int totalCount = 2 * nodes - 1;
	PixelInfo* pixelInfo = new PixelInfo[totalCount];
	for (int i = 0; i < totalCount; ++i) pixelInfo[i].setSize(maxLen);
	HuffCode* huffCode = new HuffCode[totalCount];

	int j = 0;
	double temp;
	int pixelCount = input.rows * input.cols;
	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0) {
			huffCode[j].intens = i;
			pixelInfo[j].intens = i;

			huffCode[j].arrloc = j;

			temp = static_cast<double>(hist[i]) / static_cast<double>(pixelCount);
			pixelInfo[j].prob = temp;
			huffCode[j].prob = temp;

			pixelInfo[j].left = NULL;
			pixelInfo[j].right = NULL;

			pixelInfo[j].code[0] = '\0';
			++j;
		}
	}

	HuffCode tempHuff;
	for (int i = 0; i < nodes; ++i)
		for (j = i + 1; j < nodes; ++j)
			if (huffCode[i].prob < huffCode[j].prob) {
				tempHuff = huffCode[i];
				huffCode[i] = huffCode[j];
				huffCode[j] = tempHuff;
			}

	double sumprob;
	int sumintens;
	int n = 0, k = 0;
	int nextnode = nodes;
	while (n < nodes - 1) {
		sumprob = huffCode[nodes - n - 1].prob + huffCode[nodes - n - 2].prob;
		sumintens = huffCode[nodes - n - 1].intens + huffCode[nodes - n - 2].intens;

		pixelInfo[nextnode].intens = sumintens;
		pixelInfo[nextnode].prob = sumprob;
		pixelInfo[nextnode].left = &pixelInfo[huffCode[nodes - n - 2].arrloc];
		pixelInfo[nextnode].right = &pixelInfo[huffCode[nodes - n - 1].arrloc];
		pixelInfo[nextnode].code[0] = '\0';

		int i = 0;
		while (sumprob <= huffCode[i].prob) ++i;
		for (k = nodes; k >= 0; --k) {
			if (k == i) {
				huffCode[k].intens = sumintens;
				huffCode[k].prob = sumprob;
				huffCode[k].arrloc = nextnode;
			}
			else if (k > i)
				huffCode[k] = huffCode[k - 1];
		}
		++n;
		++nextnode;
	}

	char left = '0';
	char right = '1';
	int index;
	for (int i = totalCount - 1; i >= nodes; --i) {
		if (pixelInfo[i].left != NULL)
			strconcat(pixelInfo[i].left->code, pixelInfo[i].code, left);
		if (pixelInfo[i].right != NULL)
			strconcat(pixelInfo[i].right->code, pixelInfo[i].code, right);
	}

	for (int i = 0; i < nodes; ++i) {
		std::cout << huffCode[i].intens << " -- " << pixelInfo[i].code;
	}

	delete[] pixelInfo;
	delete[] huffCode;
}
void GrowFilter(cv::Mat& image, int thr) {
	Matrix A(image.rows, std::vector<int>(image.cols));
	int count = 0;
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j) {
			cv::Vec3b color = image.at<cv::Vec3b>(i, j);
			int I = ToGray(color);

			int I_left = 1000;
			int I_up = 1000;

			if (i > 0) {
				cv::Vec3b tmp_color = image.at<cv::Vec3b>(i - 1, j);
				I_left = ToGray(tmp_color);
			}
			if (j > 0) {
				cv::Vec3b tmp_color = image.at<cv::Vec3b>(i, j - 1);
				I_up = ToGray(tmp_color);
			}

			if (abs(I - I_left) > thr&& abs(I - I_up) > thr) {
				count++;
				image.at<cv::Vec3b>(i, j)[0] = I;
				image.at<cv::Vec3b>(i, j)[1] = I;
				image.at<cv::Vec3b>(i, j)[2] = I;
				A[i][j] = count;
			}
			else if (abs(I - I_left) < thr && abs(I - I_up) < thr && I_left != I_up) {
				if (abs(I_up - I_left) < thr) {
					count++;
					int min = 0;
					Merge(image, A[i - 1][j], A[i][j - 1], I, i, j, A, count);
					image.at<cv::Vec3b>(i, j)[0] = I;
					image.at<cv::Vec3b>(i, j)[1] = I;
					image.at<cv::Vec3b>(i, j)[2] = I;
					A[i][j] = count;
				}
				else {
					if (abs(I - I_left) < abs(I - I_up)) {
						image.at<cv::Vec3b>(i, j)[0] = I_left;
						image.at<cv::Vec3b>(i, j)[1] = I_left;
						image.at<cv::Vec3b>(i, j)[2] = I_left;
						A[i][j] = A[i - 1][j];
					}
					else {
						image.at<cv::Vec3b>(i, j)[0] = I_up;
						image.at<cv::Vec3b>(i, j)[1] = I_up;
						image.at<cv::Vec3b>(i, j)[2] = I_up;
						A[i][j] = A[i][j - 1];
					}
				}
			}
			else if (abs(I - I_left) < thr) {
				image.at<cv::Vec3b>(i, j)[0] = I_left;
				image.at<cv::Vec3b>(i, j)[1] = I_left;
				image.at<cv::Vec3b>(i, j)[2] = I_left;
				A[i][j] = A[i - 1][j];
			}
			else if (abs(I - I_up) < thr) {
				image.at<cv::Vec3b>(i, j)[0] = I_up;
				image.at<cv::Vec3b>(i, j)[1] = I_up;
				image.at<cv::Vec3b>(i, j)[2] = I_up;
				A[i][j] = A[i][j - 1];
			}

		}
}


//=============================================================
int main()
{

	setlocale(LC_ALL, "rus");

	Mat image = imread("lena.png");
	Mat FiltredImage;
	image.copyTo(FiltredImage);
	imshow("image", image);

	Mat wl1;
	Mat wl2;
	Mat wl3;
	Mat dl1;
	Mat dl2;
	Mat dl3;

	double porog = 0.5;
	cout << "Коэфф порога: " << porog << endl;

	if (image.channels() > 1)
		cvtColor(image, image, COLOR_BGR2GRAY);

	double** data = new double* [image.rows];

	for (int i = 0; i < image.rows; i++)
	{
		data[i] = new double[image.cols];
	}


	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			data[i][j] = (int)image.at<uchar>(i, j);


	WaveletRows(data, image.rows, image.cols);
	SeparateRows(data, image.rows, image.cols, 1);
	WaveletCols(data, image.rows, image.cols);
	SeparateCols(data, image.rows, image.cols, 1);
	wl1 = DataToMat(data, image.rows, image.cols);
	imshow("wavelet level 1", wl1);
	//==========================================================

	WaveletRows(data, image.rows / 2, image.cols / 2);
	SeparateRows(data, image.rows, image.cols, 2);
	WaveletCols(data, image.rows / 2, image.cols / 2);
	SeparateCols(data, image.rows, image.cols, 2);
	wl2 = DataToMat(data, image.rows, image.cols);
	imshow("wavelet level 2", wl2);

	//==========================================================

	WaveletRows(data, image.rows / 4, image.cols / 4);
	SeparateRows(data, image.rows, image.cols, 4);
	WaveletCols(data, image.rows / 4, image.cols / 4);
	SeparateCols(data, image.rows, image.cols, 4);
	wl3 = DataToMat(data, image.rows, image.cols);
	imshow("wavelet level 3", wl3);

	cout << "Количество обнуленных пикселей: " << HowMuchThreshold(data, image.rows, image.cols, porog) << endl;

	DeleteLessThreshold(data, image.rows, image.cols, porog);

	//==========================================================

	BackSeparateCols(data, image.rows, image.cols, 4);
	BackWaveletCols(data, image.rows / 4, image.cols / 4);
	BackSeparateRows(data, image.rows, image.cols, 4);
	BackWaveletRows(data, image.rows / 4, image.cols / 4);
	dl3 = DataToMat(data, image.rows, image.cols);
	imshow("decoder level 3", dl3);

	//decod==========================================================

	BackSeparateCols(data, image.rows, image.cols, 2);
	BackWaveletCols(data, image.rows / 2, image.cols / 2);
	BackSeparateRows(data, image.rows, image.cols, 2);
	BackWaveletRows(data, image.rows / 2, image.cols / 2);
	dl2 = DataToMat(data, image.rows, image.cols);
	imshow("decoder level 2", dl2);

	//==========================================================

	BackSeparateCols(data, image.rows, image.cols, 1);
	BackWaveletCols(data, image.rows, image.cols);
	BackSeparateRows(data, image.rows, image.cols, 1);
	BackWaveletRows(data, image.rows, image.cols);
	dl1 = DataToMat(data, image.rows, image.cols);
	imshow("decoder level 1", dl1);

	//==========================================================

	imwrite("C:\\App\\greypicture.png", wl3);
	imwrite("C:\\App\\grey.png", image);

	//============================================

	double loss = (double)((double)(Difference(image, dl1) * 100)) / (image.rows * image.cols);
	cout << "Процент потерь: " << loss << endl;

	//============================================

	waitKey();
	return 0;
}
