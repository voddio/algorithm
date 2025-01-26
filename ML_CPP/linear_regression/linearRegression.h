# ifndef LINEAR_REGRESSION_H
# define LINEAR_REGRESSION_H

# include <vector>


class linearRegression
{
    public:
        linearRegression(int num_features);
        ~linearRegression();
        void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y, int epochs, int batch_size, double learning_rate);
        double predict(const std::vector<double> &x);
        std::vector<double> predict(const std::vector<std::vector<double>> &X);
        void print_params();
    private:
        std::vector<double> w;
        double b;
        double cal_loss(const std::vector<std::vector<double>> &X, const std::vector<double>& y);
};
# endif