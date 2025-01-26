# include "linearRegression.h"
# include "utils.h"
# include <iostream>
# include <cmath>
# include <random>
# include <algorithm>
# include <cassert>

linearRegression::linearRegression(int num_features)
{
    w.resize(num_features, 0.0);
    b = 0.0;
}

linearRegression::~linearRegression()
{
}

void linearRegression::train(const std::vector<std::vector<double>> &X, const std::vector<double> &y, int epochs, int batch_size, double learning_rate)
{
    int N_samples = X.size();
    int n_features = X[0].size();
    std::vector<double> grad_w(n_features, 0.0);
    double grad_b = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::vector<int> indices(N_samples);
        for (int i = 0; i < N_samples; i++)
            indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

        // calculate gradient for mini-batch
        for (int start = 0; start < N_samples; start += batch_size)
        {
            // every mini-batch dw, db need to reset 0
            grad_b = 0.0;
            std::fill(grad_w.begin(), grad_w.end(), 0.0);

            // dw = sum((w*x_i + b - y_i) * x_i)
            for (int i = start; i < std::min(start + batch_size, N_samples); i++)
            {
                int idx = indices[i];
                double y_pred = predict(X[idx]);
                double err = y_pred - y[idx];
                
                for (int j = 0; j < n_features; j++)
                {
                    grad_w[j] += err * X[idx][j];
                }
                grad_b += err;
            }
            // update w, b
            for (int j = 0; j < n_features; j++)
            {
                w[j] -= (learning_rate / batch_size) * grad_w[j];
            }
            b -= (learning_rate / batch_size) * grad_b;
        }
        
        
        if (epoch % 10 == 0)
        {
            std::cout<< "Epoch [" << epoch << "/" << epochs << "], Loss: " << cal_loss(X, y) << std::endl;
        }
    }
}

double linearRegression::cal_loss(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
{
    double loss = 0.0;
    int N_samples = X.size();
    for (int i = 0; i < N_samples; i++)
    {
        double y_pred = predict(X[i]);
        loss += (y_pred - y[i]) * (y_pred - y[i]);
    }
    return loss / (2 * N_samples);
}

double linearRegression::predict(const std::vector<double> &x)
{
    double res = b;
    int n_features = x.size();
    for (int j = 0; j < n_features; j++)
    {
        res += w[j] * x[j];
    }
    return res;
}

std::vector<double> linearRegression::predict(const std::vector<std::vector<double>> &X)
{
    std::vector<double> y_pred(X.size(), b);
    for (int i = 0; i < X.size(); i++)
    {
        for (int j = 0; j < X[0].size(); j++)
        {
            y_pred[i] += w[j] * X[i][j];
        }
    }
    return y_pred;
}

void linearRegression::print_params()
{
    std::cout << "Model Params: " << std::endl;
    std::cout << "Bias: " << b << std::endl;
    for (int i = 0; i < w.size(); i++)
    {
        std::cout << "Weight " << i + 1 << ": " << w[i] << std::endl;
    }
}