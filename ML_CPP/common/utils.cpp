# include "utils.h"
# include <fstream>
# include <sstream>
# include <iostream>
# include <cmath>
# include <random>
# include <algorithm>


bool load_data(const std::string& filename, std::vector<std::vector<double>> &data)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open data file." << std::endl;
        return false; 
    }
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
        row.clear();
    }
    return true;
}

// TODO
void train_test_split(const std::vector<std::vector<double>> &data, 
                            std::vector<std::vector<double>> &X_train, 
                            std::vector<double> &y_train, 
                            std::vector<std::vector<double>> &X_test, 
                            std::vector<double> &y_test,
                            const float test_size,
                            bool shuffle)
{
    int N_samples = data.size();
    std::vector<int> indices(N_samples);
    for (int i = 0; i < N_samples; i++)
    {
        indices[i] = i;
    }
    if (shuffle)
    {
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    }
    int Ntrain = ceil((1 - test_size) * N_samples);
    if (test_size <= 0 or test_size >= 1)
    {
        std::cout << "test_size must in (0, 1)!" << std::endl;
        return;
    }
    for (int i = 0; i < Ntrain; i++)
    {
        y_train.push_back(data[indices[i]].back());
        X_train.push_back(std::vector<double>(data[indices[i]].begin(), data[indices[i]].end() - 1));
    }
    for (int i = Ntrain; i < N_samples; i++)
    {
        y_test.push_back(data[indices[i]].back());
        X_test.push_back(std::vector<double>(data[indices[i]].begin(), data[indices[i]].end() - 1));
    }
    return;
}

// std::vector<std::vector<double>> MinMaxScale(const std::vector<std::vector<double>> &data)
// {
//     int N_samples = data.size();
//     if (N_samples <= 1)
//     {
//         std::cout << "[Warning] Samples number: " << N_samples << std::endl;
//         return data;
//     }
//     int N_features = data[0].size();
//     std::vector<std::vector<double>> res(N_samples, std::vector<double>(N_features));

//     for (int j = 0; j < N_features; j++)
//     {
//         double cur_max = data[0][j];
//         double cur_min = data[0][j];
//         for (int i = 1; i < N_samples; i++)
//         {
//             cur_max = std::max(cur_max, data[i][j]);
//             cur_min = std::min(cur_min, data[i][j]);
//         }
//         double data_range = cur_max - cur_min;
//         if (data_range == 0)
//         {
//             std::cout << "[Warning] all nums in Col: " << j << " = 0!" << std::endl;
//             for (int i = 0; i < N_samples; i++)
//             {
//                 res[i][j] = 0;
//             }
//         }
//         else
//         {
//             for (int i = 0; i < N_samples; i++)
//             {
//                 res[i][j] = (data[i][j] - cur_min) / data_range;
//             }
//         }
//     }
//     return res;
// }

MinMaxScaler::MinMaxScaler() {}

MinMaxScaler::~MinMaxScaler() {}

void MinMaxScaler::fit(const std::vector<std::vector<double>> data)
{
    int N_samples = data.size();
    int N_features = data[0].size();
    min_.resize(N_features, 0.0);
    max_.resize(N_features, 0.0);
    for (int j = 0; j < N_features; j++)
    {
        double cur_max = data[0][j];
        double cur_min = data[0][j];
        for (int i = 1; i < N_samples; i++)
        {
            cur_max = std::max(cur_max, data[i][j]);
            cur_min = std::min(cur_min, data[i][j]);
        }
        max_[j] = cur_max;
        min_[j] = cur_min;
    }
}

std::vector<std::vector<double>> MinMaxScaler::transform(const std::vector<std::vector<double>> data)
{
    if (max_.size() == 0)
    {
        std::cout << "[Error]Please fit the scaler first!" << std::endl;
        return data;
    }
    int N_samples = data.size();
    if (N_samples <= 1)
    {
        std::cout << "[Warning] Samples number: " << N_samples << std::endl;
        return data;
    }
    int N_features = data[0].size();
    std::vector<std::vector<double>> res(N_samples, std::vector<double>(N_features));

    for (int j = 0; j < N_features; j++)
    {
        double data_range = max_[j] - min_[j];
        if (data_range == 0)
        {
            std::cout << "[Warning] all nums in Col: " << j << " are equal!" << std::endl;
            for (int i = 0; i < N_samples; i++)
            {
                res[i][j] = 0;
            }
        }
        else
        {
            for (int i = 0; i < N_samples; i++)
            {
                res[i][j] = (data[i][j] - min_[j]) / data_range;
            }
        }
    }
    return res;
}

std::vector<std::vector<double>> MinMaxScaler::fit_transform(const std::vector<std::vector<double>> data)
{
    int N_samples = data.size();
    if (N_samples <= 1)
    {
        std::cout << "[Warning] Samples number: " << N_samples << std::endl;
        return data;
    }
    int N_features = data[0].size();
    std::vector<std::vector<double>> res(N_samples, std::vector<double>(N_features));

    for (int j = 0; j < N_features; j++)
    {
        double cur_max = data[0][j];
        double cur_min = data[0][j];
        for (int i = 1; i < N_samples; i++)
        {
            cur_max = std::max(cur_max, data[i][j]);
            cur_min = std::min(cur_min, data[i][j]);
        }
        max_[j] = cur_max;
        min_[j] = cur_min;
        double data_range = cur_max - cur_min;
        if (data_range == 0)
        {
            std::cout << "[Warning] all nums in Col: " << j << " are equal." << std::endl;
            for (int i = 0; i < N_samples; i++)
            {
                res[i][j] = 0;
            }
        }
        else
        {
            for (int i = 0; i < N_samples; i++)
            {
                res[i][j] = (data[i][j] - cur_min) / data_range;
            }
        }
    }
    return res;
}

void MinMaxScaler::print_params()
{
    std::cout << "Max_: " << std::endl;
    print_matrix(max_);
    std::cout << "Min_: " << std::endl;
    print_matrix(min_);
}


std::pair<std::vector<std::vector<double>>, std::vector<double>> 
generate_data(int n, double mean_x, double mean_y, double stddev, int label)
{
    std::vector<std::vector<double>> data;
    std::vector<double> labels;
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist_x(mean_x, stddev);
    std::normal_distribution<double> dist_y(mean_y, stddev);

    for(int i = 0; i < n; i++) {
        data.push_back({dist_x(rng), dist_y(rng)});
        labels.push_back(label);
    }
    return {data, labels};
}

void save_data(const std::vector<std::vector<double>>& X, const std::vector<double>& y) 
{
    std::ofstream x_file("X.csv"), y_file("y.csv");
    for (const auto& sample : X) 
    {
        x_file << sample[0] << "," << sample[1] << "\n";
    }
    for (auto label : y) 
    {
        y_file << label << "\n";
    }
    x_file.close();
    y_file.close();
}