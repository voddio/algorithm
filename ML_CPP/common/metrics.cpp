# include "metrics.h"
# include <vector>
# include <iostream>
# include <cmath>
# include <numeric>


double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    // R2 = 1 - sum (y^hat - y)^2 / sum (y - y_bar)^2
    int N_samples = y_true.size();
    if ((N_samples != y_pred.size()) or N_samples == 0)
    {
        std::cout << "y_true size: " << N_samples << "y_pred size: " << y_pred.size() << std::endl;
        return -1;
    }

    double y_bar = std::accumulate(y_true.begin(), y_true.end(), 0.0) / N_samples;;
    double SS_tol = 0.0;
    double SS_res = 0.0;
    for (int i = 0; i < N_samples; i++)
    {
        SS_tol += (y_true[i] - y_bar) * (y_true[i] - y_bar);
        SS_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    if (SS_tol == 0)
    {
        std::cout << "All nums in y_true are same!" << std::endl;
        return -1;
    }
    return 1 - SS_res / SS_tol;
}


double MAE(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    // MAE = sum abs(y_hat - yi) / N
    int N_samples = y_true.size();
    if ((N_samples != y_pred.size()) or N_samples == 0)
    {
        std::cout << "y_true size: " << N_samples << "y_pred size: " << y_pred.size() << std::endl;
        return -1;
    }
    double sums = 0.0;
    for (int i = 0; i < N_samples; i++)
    {
        sums += std::abs(y_pred[i] - y_true[i]);
    }
    return sums / N_samples;
}


double MSE(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    // MSE = sum (y_hat - yi)**2 / N
    int N_samples = y_true.size();
    if ((N_samples != y_pred.size()) or N_samples == 0)
    {
        std::cout << "y_true size: " << N_samples << "y_pred size: " << y_pred.size() << std::endl;
        return -1;
    }
    double sums = 0.0;
    for (int i = 0; i < N_samples; i++)
    {
        sums += (y_pred[i] - y_true[i]) * (y_pred[i] - y_true[i]);
    }
    return sums / N_samples;
}


double RMSE(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    // rmse = sqrt(MSE)
    double mse = MSE(y_true, y_pred);
    if (mse == -1)
    {
        return -1;
    }
    return std::sqrt(mse);
}

double MAPE(const std::vector<double>& y_true, const std::vector<double>& y_pred)
{
    // MAPE = SUM abs((y_hat - yi)/yi) /N
    int N_samples = y_true.size();
    if ((N_samples != y_pred.size()) or N_samples == 0)
    {
        std::cout << "y_true size: " << N_samples << "y_pred size: " << y_pred.size() << std::endl;
        return -1;
    }
    double sums = 0.0;
    for (int i = 0; i < N_samples; i++)
    {
        if (y_true[i] == 0)
            continue;
        sums += std::abs((y_pred[i] - y_true[i]) / y_true[i]);
    }
    return sums / N_samples;
}

