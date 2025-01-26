# include <iostream>
# include <Python.h>
# include <string>
# include <vector>
# include "linearRegression.h"
# include "svm.h"
# include "utils.h"
# include "metrics.h"

// # define WITHOUT_NUMPY
# include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

void test_SVM_model();
void test_SVM_model2();
int main()
{
    Py_SetPythonHome(L"D:\\Users\\44225\\anaconda3"); 
    // std::string filename = "winequality-white.csv";

    // std::vector<std::vector<double>> data;
    // if (!load_data(filename, data))
    //     return -1;
    // std::vector<std::vector<double>> X_train;
    // std::vector<double> y_train;
    // std::vector<std::vector<double>> X_test;
    // std::vector<double> y_test;
    // train_test_split(data, X_train, y_train, X_test, y_test, 0.3, false);

    // MinMaxScaler sc;
    // sc.fit(X_train);

    // std::vector<std::vector<double>> X_train_norm;
    // std::vector<std::vector<double>> X_test_norm;
    // X_train_norm = sc.transform(X_train);
    // X_test_norm = sc.transform(X_test);

    // std::cout << "Train:" << std::endl;
    // print_matrix(X_train_norm);
    // std::cout << "Test:" << std::endl;
    // print_matrix(X_test_norm);

    // test_LR_model(X_train_norm, y_train, X_test_norm, y_test);

    test_SVM_model2();
}

void test_lr_model(const std::vector<std::vector<double>> X_train_norm,
                    const std::vector<double> y_train,
                    const std::vector<std::vector<double>> X_test_norm,
                    const std::vector<double> y_test)
{
    linearRegression lr_model(X_train_norm[0].size());
    lr_model.train(X_train_norm, y_train, 200, 1000, 0.01);

    // lr_model.print_params();
    std::vector<double> y_pred;
    for (auto xi : X_test_norm)
        y_pred.push_back(lr_model.predict(xi));
    double r2, mae, mse, rmse, mape;
    r2 = r2_score(y_test, y_pred);
    mae = MAE(y_test, y_pred);
    mse = MSE(y_test, y_pred);
    rmse = RMSE(y_test, y_pred);
    mape = MAPE(y_test, y_pred);

    std::cout << "R2: " << r2 << std::endl;
    std::cout << "MAE: " << mae << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    std::cout << "MAPE: " << mape << std::endl;

    plt::scatter(y_test, y_pred);
    plt::xlabel("True val");
    plt::ylabel("Pred val");
    plt::show();
    plt::save("output.png");
    return;
}

// void test_SVM_model(const std::vector<std::vector<double>> X_train_norm,
//                     const std::vector<double> y_train,
//                     const std::vector<std::vector<double>> X_test_norm,
//                     const std::vector<double> y_test)
void test_SVM_model()
{
    SVM svm_model(1.0, 0.001, "linear", 0.5, 10, 3, 1.0);

    std::vector<std::vector<double>> X = {{1, 2}, {2, 3}, {3, 3}, {6, 5}, {7, 8}, {8, 9}};
    std::vector<double> y = {1, 1, 1, -1, -1, -1};
    print_matrix(X);

    svm_model.train(X, y);
    auto support_vectors = svm_model.get_support_vectors();
    print_matrix(support_vectors);


    std::vector<double> xx, yy, zz;
    for (double x = 0; x < 10; x += 0.1) {
        for (double y = 0; y < 10; y += 0.1) {
            xx.push_back(x);
            yy.push_back(y);
            std::vector<double> xi = {x, y};
            zz.push_back(svm_model.predict(xi)); 
        }
    }

    std::vector<double> pos_x, pos_y, neg_x, neg_y;
    for (size_t i = 0; i < X.size(); ++i) {
        if (y[i] == 1) {
            pos_x.push_back(X[i][0]);
            pos_y.push_back(X[i][1]);
        } else {
            neg_x.push_back(X[i][0]);
            neg_y.push_back(X[i][1]);
        }
    }

    plt::scatter(pos_x, pos_y, 10, {{"color", "red"}});
    plt::scatter(neg_x, neg_y, 10, {{"color", "blue"}});

    std::vector<double> sv_x, sv_y;
    for (const auto& sv : support_vectors) {
        sv_x.push_back(sv[0]);
        sv_y.push_back(sv[1]);
    }
    plt::scatter(sv_x, sv_y, 50, {{"color", "green"}, {"marker", "x"}});

    plt::title("SVM Decision Boundary");
    plt::xlabel("Feature 1");
    plt::ylabel("Feature 2");

    plt::show();
    return;
}



void test_SVM_model2()
{
    auto [pos_data, pos_labels] = generate_data(10, 1.0, 1.0, 0.6, 1);
    auto [neg_data, neg_labels] = generate_data(10, -1.0, -1.0, 0.6, -1);

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    X.insert(X.end(), pos_data.begin(), pos_data.end());
    X.insert(X.end(), neg_data.begin(), neg_data.end());
    y.insert(y.end(), pos_labels.begin(), pos_labels.end());
    y.insert(y.end(), neg_labels.begin(), neg_labels.end());

    save_data(X, y);

    SVM svm(2.0, 0.001, "rbf", 0.5, 50, 3, 1.0);
    svm.train(X, y);

    const double x_min = -3.0, x_max = 3.0;
    const double y_min = -3.0, y_max = 3.0;
    const double step = 0.1;
    std::vector<std::vector<double>> X_grid, Y_grid;
    std::vector<std::vector<double>> Z_grid;
    for(double x = x_min; x <= x_max; x += step) {
        std::vector<double> x_row, y_row, z_row;
        for(double y = y_min; y <= y_max; y += step) {
            x_row.push_back(x);
            y_row.push_back(y);
            z_row.push_back(svm.predict({x, y}));
        }
        X_grid.push_back(x_row);
        Y_grid.push_back(y_row);
        Z_grid.push_back(z_row);
    }

    std::vector<double> pos_x, pos_y, neg_x, neg_y;
    for(size_t i = 0; i < y.size(); i++) 
    {
        if(y[i] > 0) 
        {
            pos_x.push_back(X[i][0]);
            pos_y.push_back(X[i][1]);
        } else 
        {
            neg_x.push_back(X[i][0]);
            neg_y.push_back(X[i][1]);
        }
    }
    plt::contour(X_grid, Y_grid, Z_grid, {{"cmap", "coolwarm"}});
    plt::scatter(pos_x, pos_y, 30, {{"color", "red"}, {"label", "Positive"}});
    plt::scatter(neg_x, neg_y, 30, {{"color", "blue"}, {"label", "Negative"}});
    
    auto support_vectors = svm.get_support_vectors();
    std::cout << "support_vectors: " << std::endl;
    print_matrix(support_vectors);
    std::vector<double> sv_x, sv_y;
    for(const auto& sv : support_vectors) 
    {
        sv_x.push_back(sv[0]);
        sv_y.push_back(sv[1]);
    }

    plt::scatter(sv_x, sv_y, 50, {{"color", "black"}, {"marker", "x"}, {"label", "Support Vectors"}});
    plt::title("SVM Decision Boundary");
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::xlim(x_min, x_max);
    plt::ylim(y_min, y_max);
    plt::legend();
    plt::show();

    return;
}