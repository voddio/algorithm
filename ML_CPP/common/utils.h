# ifndef UTILS_H
# define UTILS_H

# include <vector>
# include <string>
# include <iomanip>
# include <iostream>

bool load_data(const std::string& filename, std::vector<std::vector<double>> &X);
std::pair<std::vector<std::vector<double>>, std::vector<double>> 
generate_data(int n, double mean_x, double mean_y, double stddev, int label);
void save_data(const std::vector<std::vector<double>>& X, const std::vector<double>& y);


template <typename T>
void print_matrix(const std::vector<std::vector<T>>& mat)
{
    if (mat.size() == 0)
        std::cout << "The matrix is NULL" << std::endl;
    for (int i = 0; i < mat.size(); i++)
    {
        for (int j = 0; j < mat[0].size(); j++)
        {
            std::cout << std::setw(8) << mat[i][j] << "\t"; 
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_matrix(const std::vector<T>& mat)
{
    if (mat.size() == 0)
        std::cout << "The matrix is NULL" << std::endl;
    for (int i = 0; i < mat.size(); i++)
    {
        std::cout << std::setw(8) << mat[i] << "\t"; 
    }
    std::cout << std::endl;
}

void train_test_split(const std::vector<std::vector<double>> &data, 
                            std::vector<std::vector<double>> &X_train, 
                            std::vector<double> &y_train, 
                            std::vector<std::vector<double>> &X_test, 
                            std::vector<double> &y_test,
                            const float test_size = 0.3,
                            bool shuffle = true);

// std::vector<std::vector<double>> MinMaxScale(const std::vector<std::vector<double>> &data);
class MinMaxScaler
{
    public:
        MinMaxScaler();
        ~MinMaxScaler();
        std::vector<double> max_;
        std::vector<double> min_;
        void fit(const std::vector<std::vector<double>> data);
        std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> data);
        std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>> data);
        void print_params();
};

template <typename T>
T max(const T a, const T b)
{
    return a > b ? a : b;
}

template <typename T>
T min(const T a, const T b)
{
    return a < b ? a : b;
}

template <typename T>
T dot(const std::vector<T> &xi, const std::vector<T> &xj)
{
    T temp = 0;
    for (int i = 0; i < xi.size(); i++)
    {
        temp += xi[i] * xj[i];
    }
    return temp;
}

template <typename T>
std::vector<T> minus(const std::vector<T> &xi, const std::vector<T> &xj)
{
    std::vector<T> temp (xi.size(), 0);
    for (int i = 0; i < xi.size(); i++)
    {
        temp[i] = xi[i] - xj[i];
    }
    return temp;
}
# endif